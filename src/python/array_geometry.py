"""Authoritative ITRF metre coordinates and a shared local ENU reference frame."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation


def itrf_to_enu(positions_itrf_m, reference_itrf_m=None):
    xyz = np.asarray(positions_itrf_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('ITRF coordinates must be finite metres [N,3]')
    radius = np.linalg.norm(xyz, axis=1)
    if np.any((radius < 6e6) | (radius > 7e6)):
        raise ValueError('ITRF coordinates must be geocentric metres near the Earth surface')
    origin = xyz.mean(axis=0) if reference_itrf_m is None else np.asarray(reference_itrf_m, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all() or not 6e6 <= np.linalg.norm(origin) <= 7e6:
        raise ValueError('invalid ITRF reference point')
    site = EarthLocation.from_geocentric(*origin, unit=u.m)
    lon, lat = site.lon.rad, site.lat.rad
    sl, cl, sp, cp = np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)
    rotation = np.array([[-sl,cl,0],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]])
    return dict(positions_m=(xyz-origin)@rotation.T, reference_itrf_m=origin,
                itrf_to_enu_rotation=rotation, site_lon_deg=float(site.lon.deg),
                site_lat_deg=float(site.lat.deg), site_height_m=float(site.height.to_value(u.m)))


def ordered_antenna_names(index_map):
    if not index_map:
        raise ValueError('empty antenna_to_epic_index')
    values = list(index_map.values())
    if any(isinstance(i, bool) or not isinstance(i, int) for i in values) or sorted(values) != list(range(len(values))):
        raise ValueError('antenna indices must be unique contiguous integers starting at zero')
    return [name for name, index in sorted(index_map.items(), key=lambda item: item[1])]


def load_itrf_array(path, antenna_names):
    """Reference = centroid of all physical rows, independent of selected inputs/PADs.

    Legacy projected/geodetic columns are deliberately not used. PAD inputs are
    placed at the reference point; their voltages must remain zero.
    """
    path = Path(path)
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            name = row.get('名称') or row.get('name') or row.get('Name')
            if not name:
                continue
            if name in rows:
                raise ValueError(f'duplicate antenna name in coordinates: {name}')
            if name.startswith('PAD'):
                raise ValueError('coordinate survey must contain physical antennas only')
            try:
                rows[name] = [float(row['ITRF_'+axis]) for axis in 'XYZ']
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{name}: finite ITRF_X/Y/Z metre columns are required') from exc
    names = list(antenna_names)
    if not names or len(set(names)) != len(names) or not any(not n.startswith('PAD') for n in names):
        raise ValueError('unique antenna names with at least one physical input are required')
    # Sort rows so changing CSV order cannot change the reference or its rounding.
    physical_names = sorted(rows)
    geometry = itrf_to_enu([rows[name] for name in physical_names])
    by_name = dict(zip(physical_names, geometry['positions_m']))
    positions = []
    for name in names:
        if name.startswith('PAD'):
            positions.append(np.zeros(3))
        elif name in by_name:
            positions.append(by_name[name])
        else:
            raise ValueError(f'{name}: missing authoritative ITRF coordinates')
    geometry.update(positions_m=np.array(positions), antenna_names=names,
                    coordinate_frame='local_ENU', coordinate_authority='ITRF_X/Y/Z',
                    reference_policy='centroid_of_all_physical_CSV_rows',
                    coordinate_source=str(path.resolve()), coordinate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return geometry


def geometry_metadata(geometry):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in geometry.items() if key != 'positions_m'}
