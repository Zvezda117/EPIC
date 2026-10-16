import numpy as np
from lsl.common.stations import lwasv
from astropy.constants import c as speed_of_light
import matplotlib.image
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
from scipy.signal import correlate2d

import datetime
import csv
import os
import time
from MCS2 import Communicator
from pathlib import Path
import uuid
from daily_digest import ingest_daily_digest

DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"
FS = 196.0e6
CHAN_BW = 25.0e3
ADP_EPOCH = datetime.datetime(1970, 1, 1)

lwasv_station = lwasv


def _candidate_repo_paths(*parts):
    roots = [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parents[2]]
    seen = set()
    for root in roots:
        path = (root / Path(*parts)).resolve()
        if path in seen:
            continue
        seen.add(path)
        yield path


def _find_repo_file(env_name, default_name):
    env_path = os.environ.get(env_name)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"{env_name} points to a missing file: {path}")
    for path in _candidate_repo_paths(default_name):
        if path.is_file():
            return path
    return None


def _read_simple_yaml(path):
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read antenna mapping files") from exc


def _load_coordinate_rows(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("名称") or row.get("name") or row.get("Name")
            if name:
                rows[name] = row
    return rows


def _load_custom_array(expected_nstands=None):
    mapping_path = _find_repo_file("EPIC_ARRAY_CONFIG", "antenna_mapping_9197.yaml")
    coord_path = _find_repo_file("EPIC_ANTENNA_COORDS", "antenna_coordinates.csv")
    if mapping_path is None or coord_path is None:
        return None

    mapping = _read_simple_yaml(mapping_path)
    index_map = mapping.get("antenna_to_epic_index", {})
    if not index_map:
        return None

    from array_geometry import load_itrf_array, ordered_antenna_names, geometry_metadata
    names = ordered_antenna_names(index_map)
    if expected_nstands is not None and len(names) != int(expected_nstands):
        raise RuntimeError(
            f"{mapping_path.name} defines {len(names)} stands, but this EPIC "
            f"binary was compiled for {expected_nstands}. Rebuild with matching "
            "LWA_SV_NSTANDS or select a matching EPIC_ARRAY_CONFIG."
        )
    geometry = load_itrf_array(coord_path, names)
    obs = mapping.get("observation", {})
    return {
        "name": mapping.get("station", {}).get("name", mapping_path.stem),
        "positions": geometry["positions_m"],
        "antenna_names": names,
        "lon_deg": geometry["site_lon_deg"],
        "lat_deg": geometry["site_lat_deg"],
        "height_m": geometry["site_height_m"],
        "geometry_metadata": geometry_metadata(geometry),
        "center_freq_mhz": obs.get("center_freq_mhz"),
    }


def _station_lon_lat_deg():
    custom = _load_custom_array()
    if custom and custom["lon_deg"] is not None and custom["lat_deg"] is not None:
        return custom["lon_deg"], custom["lat_deg"]
    return lwasv_station.lon * 180.0 / np.pi, lwasv_station.lat * 180.0 / np.pi


def _chan0_from_center_freq(nchan):
    explicit = os.environ.get("EPIC_CHAN0")
    if explicit:
        return int(round(float(explicit)))

    center_mhz = os.environ.get("EPIC_CENTER_FREQ_MHZ")
    if center_mhz is None:
        custom = _load_custom_array()
        if custom and custom.get("center_freq_mhz") is not None:
            center_mhz = custom["center_freq_mhz"]
    if center_mhz is None:
        return 0
    return int(round(float(center_mhz) * 1e6 / CHAN_BW))


def gen_loc_lwasv(grid_size, grid_resolution, nstands=None):
    """
    Generate grid centered locations of LWASV stands compatible with DFT code.
    The indices must be divided by the wavelength to transform the coordinates
    to pixel units

    grid_size: Dimension of the grid.
    grid_resolution: Grid resolution in degrees
    """

    custom = _load_custom_array(nstands)
    if custom is not None:
        lsl_locsf = custom["positions"].copy()
        names = np.asarray(custom.get("antenna_names", []), dtype=object)
        if names.size == lsl_locsf.shape[0]:
            pad_mask = np.asarray([str(name).startswith("PAD") for name in names])
            if np.any(pad_mask):
                real_mask = ~pad_mask
                if not np.any(real_mask):
                    raise RuntimeError("custom array contains only padding stands")
                # Padding stands are zero-valued data placeholders used to satisfy
                # EPIC's compiled input width. Keep them near the array centre so
                # they do not expand the min/max coordinate span and push the real
                # L-band antennas off the image grid.
                lsl_locsf[pad_mask] = np.mean(lsl_locsf[real_mask], axis=0)
    else:
        lsl_locsf = np.array(
            [(ant.stand.x, ant.stand.y, ant.stand.z) for ant in lwasv[::2]]
        )
        lsl_locsf[
            [i for i, a in enumerate(lwasv[::2]) if a.stand.id == 256], :
        ] = 0.0

    # adjusted to ensure the pixel size <= half the wavelength
    delta = (2 * grid_size * np.sin(np.pi * grid_resolution / 360)) ** -1
    lsl_locsf = lsl_locsf / delta
    lsl_locsf -= np.min(lsl_locsf, axis=0, keepdims=True)

    # Centre locations slightly
    # divide by wavelength and add grid_size/2 to get the correct grid
    # positions
    lsl_locsf -= np.max(lsl_locsf, axis=0, keepdims=True) / 2.0

    # add ntime axis
    # final dimensions = 3, ntime, nchan, nant, npol
    return dict(
        delta=delta, locations=lsl_locsf.astype(np.double).ravel().copy()
    )


def gen_phases_lwasv(nchan, chan0, nstands=None, npol=2):
    """
    Generate complex phases for the LWA-SV antennas

    nchan: Number of channels
    chan0: Channel number of the first channel
    """
    custom = _load_custom_array(nstands)
    if custom is not None:
        nstand = int(nstands or custom["positions"].shape[0])
        phases = np.ones((nchan, nstand, int(npol)), dtype=np.complex128)
        return phases.ravel().copy()

    nstand = int(len(lwasv.antennas) / 2)
    npol = int(npol)
    phases = np.zeros((nchan, nstand, npol), dtype=np.complex64)
    bandwidth = CHAN_BW
    freq = np.arange(chan0, chan0 + nchan) * bandwidth

    for i in range(nstand):
        # X
        a = lwasv.antennas[2 * i + 0]
        delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
        phases[:, i, 0] = np.exp(2j * np.pi * freq * delay)
        phases[:, i, 0] /= np.sqrt(a.cable.gain(freq))
        # phases[:, i, 0] = np.exp(1j * np.pi)

        # Y
        a = lwasv.antennas[2 * i + 1]
        delay = a.cable.delay(freq) - a.stand.z / speed_of_light.value
        phases[:, i, 1] = np.exp(2j * np.pi * freq * delay)
        phases[:, i, 1] /= np.sqrt(a.cable.gain(freq))
        # Explicit bad and suspect antenna masking - this will
        # mask an entire stand if either pol is bad
        if (
            lwasv.antennas[2 * i + 0].combined_status < 33
            or lwasv.antennas[2 * i + 1].combined_status < 33
        ):
            phases[:, i, :] = 0.0
        # Explicit outrigger masking - we probably want to do
        # away with this at some point
        if a.stand.id in [256]+[i for i in range(122,129)]+[185,186]:
            phases[:, i, :] = 0.0

    return phases.ravel().copy()


def save_output(output_arr, grid_size, nchan, outdir, metadata):
    """
    Save image to disk

    output_arr: Output image array
    grid_size: X or Y Size of the grid in pixels
    nchan: Number of output channlels in the image
    outdir: Name of the file to save
    metadata: Dict with the metatadata to be written to the fits file
    """
    # start = time.time()
    output_arr = output_arr.reshape((nchan, grid_size, grid_size, 4))
    # output_arr_sft = np.fft.fftshift(output_arr, axes=(2, 3))

    # print(metadata)

    # print(output_arr_sft.min(), output_arr_sft.max())
    phdu = fits.PrimaryHDU()
    for k, v in metadata.items():
        # print("py", k,v)
        phdu.header[k] = v
    phdu.header["DATE-OBS"] = Time(
        metadata["time_tag"] / FS + 1e-3 * metadata["img_len_ms"] / 2.0,
        format="unix",
        precision=6,
    ).isot

    sll = (
        2 * metadata["grid_size"] * np.sin(np.pi * metadata["grid_res"] / 360)
    ) ** -1
    from imaging_math import output_frequency_axis
    _, cfreq, delta_f = output_frequency_axis(
        metadata["chan0"], nchan, metadata.get("chan_width", CHAN_BW), CHAN_BW
    )

    phdu.header["BUNITS"] = "UNCALIB"
    phdu.header["BSCALE"] = 1e0
    phdu.header["BZERO"] = 0e0
    phdu.header["EQUINOX"] = "J2000"
    phdu.header["EXTNAME"] = "PRIMARY"
    phdu.header["GRIDDIMX"] = metadata["grid_size"]
    phdu.header["GRIDDIMY"] = metadata["grid_size"]
    phdu.header["DGRIDX"] = sll
    phdu.header["DGRIDY"] = sll
    phdu.header["INTTIM"] = metadata["img_len_ms"] * 1e-3
    phdu.header["INTTIMU"] = "SECONDS"
    phdu.header["CFREQ"] = cfreq
    phdu.header["CFREQU"] = "HZ"

    # dt = TimeDelta(1e-3 * metadata["img_len_ms"], format="sec")
    dtheta_x = 2 * np.arcsin(0.5 / (metadata["grid_size"] * sll))
    dtheta_y = 2 * np.arcsin(0.5 / (metadata["grid_size"] * sll))
    crit_pix_x = float(metadata["grid_size"] / 2 + 1)

    # Need to correct for shift in center pixel when we flipped dec dimension
    # when writing npz, Only applies for even dimension size
    crit_pix_y = (
        float(metadata["grid_size"] / 2 + 1) - (metadata["grid_size"] + 1) % 2
    )
    delta_x = -dtheta_x * 180.0 / np.pi
    delta_y = dtheta_y * 180.0 / np.pi

    crit_pix_f = (metadata["nchan"] - 1) * 0.5 + 1  # +1 for FITS numbering

    lon_deg, lat_deg = _station_lon_lat_deg()
    t0 = Time(
        metadata["time_tag"] / FS,
        format="unix",
        precision=6,
        location=(lon_deg, lat_deg),
    )

    lsts = t0.sidereal_time("apparent")
    coords = SkyCoord(
        lsts.deg, lat_deg, obstime=t0, unit="deg"
    ).transform_to(FK5(equinox="J2000"))

    # img_data = np.transpose(output_arr[:, :, :, :], (0, 1, 3, 2))
    img_data = np.transpose(output_arr[:, :, :, :], (3, 0, 2, 1))
    img_data = np.fft.fftshift(img_data, axes=(2, 3))[:, :, ::-1, :]
    # img_data = img_data / img_data.max(axis=(2, 3), keepdims=True)
    ihdu = fits.ImageHDU(img_data)

    ihdu.header["DATETIME"] = t0.isot
    ihdu.header["LST"] = lsts.hour
    ihdu.header["EQUINOX"] = "J2000"

    ihdu.header["CTYPE1"] = "RA---SIN"
    ihdu.header["CRPIX1"] = crit_pix_x
    ihdu.header["CDELT1"] = delta_x
    ihdu.header["CRVAL1"] = coords.ra.deg
    ihdu.header["CUNIT1"] = "deg"

    ihdu.header["CTYPE2"] = "DEC--SIN"
    ihdu.header["CRPIX2"] = crit_pix_y
    ihdu.header["CDELT2"] = delta_y
    ihdu.header["CRVAL2"] = coords.dec.deg
    ihdu.header["CUNIT2"] = "deg"
    # Coordinates - Freq
    ihdu.header["CTYPE3"] = "FREQ"
    ihdu.header["CRPIX3"] = crit_pix_f
    ihdu.header["CDELT3"] = delta_f
    ihdu.header["CRVAL3"] = cfreq
    ihdu.header["CUNIT3"] = "Hz"
    # # Coordinates - Stokes parameters
    ihdu.header["CTYPE4"] = "CORRPROD"
    ihdu.header["CRPIX4"] = 1
    ihdu.header["CDELT4"] = 1
    ihdu.header["CRVAL4"] = 1
    ihdu.header["POLPROD"] = "XX,YY,RE_XY,IM_XY"
    ihdu.header["POLCONV"] = "XY=X*conj(Y)"

    hdulist = fits.HDUList([phdu, ihdu])
    unix_time = metadata["time_tag"] / FS +\
     1e-3 * metadata["img_len_ms"] / 2.0
    filename = f"EPIC_{unix_time:.3f}_{cfreq/1e6:.3f}MHz"
    os.makedirs(outdir, exist_ok=True)
    fits_path = os.path.join(outdir, f"{filename}.fits")
    png_path = os.path.join(outdir, f"{filename}_QL.png")
    hdulist.writeto(fits_path, overwrite=True)

    matplotlib.image.imsave(
        png_path, np.nan_to_num(img_data[0, 0, :, :], nan=0.0), origin="lower"
    )

    # extract and ingest the pixels to DB
    # ingest_daily_digest(ihdu.header, phdu.header, img_data, cfreq)

    return fits_path


def get_ADP_time_from_unix_epoch():
    """
    Get time in seconds from the unix epoch using the UTC start timestamp from
    the ADP service
    """
    got_utc_start = False
    exception_count=0
    while not got_utc_start:
        try:
            with Communicator("MCS") as adp_control:
                utc_start = adp_control.report("UTC_START")
                # Check for valid timestamp
                utc_start_dt = datetime.datetime.strptime(
                    utc_start, DATE_FORMAT
                )
            got_utc_start = True
        except Exception as ex:
            exception_count += 1
            if exception_count > 10:
                raise ex
            time.sleep(0.1)
    # print((utc_start_dt - ADP_EPOCH).total_seconds())
    return (utc_start_dt - ADP_EPOCH).total_seconds()


def get_time_tag0_adp():
    timestamp0 = get_ADP_time_from_unix_epoch()
    time_tag0 = timestamp0 * FS
    return time_tag0


def get_time_from_unix_epoch(utcstart):
    """
    Get time difference in seconds from unix epoch

    utcstart: Timestamp string
    """
    utc_start_dt = datetime.datetime.strptime(utcstart, DATE_FORMAT)
    return (utc_start_dt - ADP_EPOCH).total_seconds()


def get_40ms_gulp(nstands=16, npols=2, expected_nseqs=None, capacity_bytes=None,
                  channel_start=0, channel_count=None, input_path=None):
    """Load quantized NPZ voltages and return the GPU's channel-major layout.

    The NPZ stays time-major on disk. Metadata keeps the logical time/channel
    shape; only the contiguous voltage buffer is transposed for CUDA.
    """
    from offline_io import load_gulp
    return load_gulp(nstands, npols, expected_nseqs, capacity_bytes,
                     channel_start, channel_count, input_path)


def gaussian(x, mu=0, sigma=4):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def integrate_pixel(fun, dx, dy, d_per_pixel=1, nsteps=5):
    sum = 0
    delta = 1 / float(nsteps)
    offset = 1 / float(2 * nsteps)

    # pragma unroll
    # xinit = offset
    # yinit = offset
    for x in range(nsteps):
        for y in range(nsteps):
            xx = x * delta + offset
            yy = y * delta + offset
            sum += fun((dx + xx) * d_per_pixel) * fun((dy + yy) * d_per_pixel)

    return sum


def get_gaussian_2D(support=3):
    g = np.zeros((support, support))
    for xx in range(support):
        for yy in range(support):
            x = xx - support / 2
            y = yy - support / 2

            g[xx, yy] = integrate_pixel(gaussian, x, y)

    g /= g.sum()
    return g


def get_correction_grid(corr_ker_arr, grid_size, support, nchan, oversample=1):
    """Return the inverse gridding power response, without shared debug files."""
    from imaging_math import correction_grid
    return correction_grid(corr_ker_arr, grid_size, support, nchan, oversample).ravel()


def get_random_uuid():
    return str(uuid.uuid1())


def meta2pgtime(time_tag0, img_len_ms):
    return str(
        Time(
            time_tag0 / FS + 1e-3 * img_len_ms / 2.0, # time tag marks the end of the sequence
            format="unix",
            precision=6,
        ).isot
    )


if __name__ == "__main__":
    freq = 1260 * 25000

    res = gen_loc_lwasv(64, 2)
    print(freq)
    print(
        32 + (res["locations"][3 * 256 : 6 * 256].astype(float) * freq / 3e8)
    )
