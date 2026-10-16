"""Time-dependent ICRS tangent-plane imaging in a declared local ENU frame.

An epoch must describe the first voltage sample in this input, not recorder startup.
No clock, cable, instrumental gain or RF-frequency calibration is inferred here.
"""
from datetime import datetime, timezone
import hashlib
import re
from pathlib import Path
from functools import lru_cache
from threading import RLock
from zoneinfo import ZoneInfo

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, CartesianRepresentation, EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta
from astropy.utils import iers

C = 299792458.0
_ASTROMETRY_LOCK = RLock()


def parse_utc(value, zone=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('input_start must be an explicit ISO first-sample timestamp')
    match = re.fullmatch(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(\.\d{1,9})?(Z|[+-]\d{2}:\d{2})?', value)
    if match is None:
        raise ValueError('input_start requires an ISO date and time with at most nine fractional digits')
    base, fraction, offset = match.groups()
    dt = datetime.fromisoformat(base + (offset or '').replace('Z', '+00:00'))
    if dt.tzinfo is None:
        if not zone:
            raise ValueError('a naive timestamp requires an explicit timezone')
        local_zone = ZoneInfo(zone)
        candidates = [dt.replace(tzinfo=local_zone, fold=fold) for fold in (0, 1)]
        valid = [candidate for candidate in candidates
                 if candidate.astimezone(timezone.utc).astimezone(local_zone).replace(tzinfo=None) == dt]
        if not valid:
            raise ValueError('nonexistent local timestamp in the specified timezone')
        if len({candidate.utcoffset() for candidate in valid}) > 1:
            raise ValueError('ambiguous local timestamp requires an explicit UTC offset')
        dt = valid[0]
    elif zone:
        # An offset and a timezone must agree; never silently reinterpret either.
        if dt.utcoffset() != dt.astimezone(ZoneInfo(zone)).utcoffset():
            raise ValueError('timestamp offset conflicts with timezone')
    epoch = Time(dt.astimezone(timezone.utc), scale='utc', precision=9)
    return epoch + TimeDelta(float(fraction or 0), format='sec')


def raw_tracking(positions_m, frequencies_hz, center):
    pos = np.asarray(positions_m, dtype=np.float64)
    freq = np.asarray(frequencies_hz, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if (pos.ndim != 2 or pos.shape[1] != 3 or freq.ndim != 1 or center.shape != (3,)
            or not all(np.isfinite(x).all() for x in (pos, freq, center))
            or (freq <= 0).any() or not np.isclose(np.linalg.norm(center), 1., atol=1e-10, rtol=0)):
        raise ValueError('raw tracking requires finite positions, positive RF frequencies and a unit center')
    return np.exp(2j * np.pi * freq[:, None] * (pos @ center)[None] / C)


class SkyModel:
    def __init__(self, config, positions_m, frequencies_hz):
        self.config = dict(config)
        if config.get('time_reference') != 'first_voltage_sample':
            raise ValueError('time_reference must be first_voltage_sample; recorder_start is insufficient')
        if config.get('coordinate_frame') != 'local_ENU':
            raise ValueError('sky geometry requires surveyed local_ENU positions')
        self.iers_table = None
        self.iers_provenance = {'source': 'astropy_local_table'}
        if config.get('iers_file'):
            table_path = Path(config['iers_file']).resolve()
            self.iers_table = iers.IERS_A.open(str(table_path))
            self.iers_provenance = dict(path=str(table_path), sha256=hashlib.sha256(table_path.read_bytes()).hexdigest())
        if self.iers_table is None:
            with _ASTROMETRY_LOCK, iers.conf.set_temp('auto_download', False):
                self.iers_table = iers.earth_orientation_table.get()
            table_path = self.iers_table.meta.get('data_path')
            if table_path and Path(table_path).is_file():
                self.iers_provenance.update(path=str(table_path), sha256=hashlib.sha256(Path(table_path).read_bytes()).hexdigest())
        self.epoch = parse_utc(config.get('input_start'), config.get('timezone'))
        pos = np.asarray(positions_m, dtype=np.float64)
        freq = np.asarray(frequencies_hz, dtype=np.float64)
        if (pos.ndim != 2 or pos.shape[1] != 3 or not len(pos)
                or freq.ndim != 1 or not len(freq) or (freq <= 0).any()
                or not np.isfinite(pos).all() or not np.isfinite(freq).all()):
            raise ValueError('finite positions[N,3] and positive frequencies[C] required')
        self.baseline = max(float(np.linalg.norm(pos - p, axis=1).max()) for p in pos)
        self.maxfreq = float(freq.max())
        self.dt = float(config['sequence_seconds'])
        self.limit = float(config.get('max_phase_drift_rad', 0.05))
        self.ra = float(config['source_ra_deg'])
        self.dec = float(config['source_dec_deg'])
        lon, lat, height = (float(config[k]) for k in ('site_lon_deg', 'site_lat_deg', 'site_height_m'))
        pixel = float(config['pixel_deg'])
        size = config['image_size']
        if (not np.isfinite([self.dt, self.limit, self.ra, self.dec, lon, lat, height, pixel]).all()
                or self.dt <= 0 or self.limit <= 0 or pixel <= 0
                or not 0 <= self.ra < 360 or not -90 <= self.dec <= 90
                or not -180 <= lon <= 180 or not -90 <= lat <= 90
                or isinstance(size, bool) or not isinstance(size, int) or size < 2):
            raise ValueError('invalid sky geometry parameters')
        self.location = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height*u.m)
        ra, dec = np.deg2rad([self.ra, self.dec])
        c = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        east = np.array([-np.sin(ra), np.cos(ra), 0.])
        north = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
        # SIN direction-cosine grid; x increases toward celestial east, y north.
        axis = (np.arange(size) - size//2) * np.deg2rad(pixel)
        l, m = np.meshgrid(axis, axis)
        if np.any(l*l+m*m >= 1):
            raise ValueError('tangent grid exceeds the SIN hemisphere')
        grid = l[..., None]*east + m[..., None]*north + np.sqrt(1-l*l-m*m)[..., None]*c
        max_step = max(np.linalg.norm(np.diff(grid, axis=i), axis=-1).max() for i in (0, 1))
        self.pixel_phase_bound = float(2*np.pi*self.maxfreq*self.baseline*max_step/C)
        if self.pixel_phase_bound >= np.pi:
            raise ValueError('pixel spacing undersamples the longest baseline (Nyquist)')
        self.sky = SkyCoord(CartesianRepresentation(grid.reshape(-1, 3).T), frame='icrs')
        self.source = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg, frame='icrs')

    @lru_cache(maxsize=4)
    def window(self, start_sequence, nseq):
        if (isinstance(start_sequence, bool) or isinstance(nseq, bool)
                or not isinstance(start_sequence, (int, np.integer))
                or not isinstance(nseq, (int, np.integer)) or start_sequence < 0 or nseq < 1):
            raise ValueError('window must have nonnegative integer start and positive integer length')
        duration = nseq*self.dt
        # Conservative baseline phase error from midpoint to either window edge.
        drift = 2*np.pi*self.maxfreq*self.baseline/C * (1.01*7.292115e-5) * duration/2
        if drift > self.limit:
            raise ValueError(f'window phase drift {drift:.6g} rad exceeds {self.limit}; shorten window-sequences')
        begin = self.epoch + TimeDelta(start_sequence*self.dt, format='sec')
        mid = begin + TimeDelta(duration/2, format='sec')
        end = begin + TimeDelta(duration, format='sec')
        # Avoid concurrent changes to Astropy configuration and implicit network I/O.
        with _ASTROMETRY_LOCK, iers.conf.set_temp('auto_download', False), \
                iers.earth_orientation_table.set(self.iers_table):
            table = iers.earth_orientation_table.get()
            _, ut1_status = table.ut1_utc(mid, return_status=True)
            _, _, pm_status = table.pm_xy(mid, return_status=True)
            if ut1_status < 0 or pm_status < 0:
                raise ValueError('observation time outside available IERS Earth-orientation data')
            frame = AltAz(obstime=mid, location=self.location, pressure=0*u.hPa)
            source = self.source.transform_to(frame)
            if source.alt.deg <= 0:
                raise ValueError('source is below the geometric horizon at this window')
            # AltAz Cartesian axes are North, East, Up. CUDA coordinates are E,N,U.
            dirs = self.sky.transform_to(frame).cartesian.xyz.value.T[:, [1, 0, 2]]
            center = source.cartesian.xyz.value[[1, 0, 2]]
        if np.any(dirs[:, 2] <= 0):
            raise ValueError('image grid crosses the geometric horizon')
        dirs.setflags(write=False)
        center.setflags(write=False)
        metadata = dict(start_utc=begin.utc.isot, midpoint_utc=mid.utc.isot, end_utc=end.utc.isot,
                        source_icrs_deg=[self.ra, self.dec], source_altaz_deg=[source.alt.deg, source.az.deg],
                        grid_projection='ICRS SIN; x east, y north; center pixel image_size//2',
                        phase_drift_bound_rad=drift, pixel_phase_bound_rad=self.pixel_phase_bound,
                        refraction='disabled', iers_table=self.iers_provenance, iers_ut1_status=int(ut1_status), iers_pm_status=int(pm_status))
        return dirs, center, metadata
