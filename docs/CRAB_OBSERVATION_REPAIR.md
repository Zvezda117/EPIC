# Crab observation-model repair — 2026-10-16

This stage repairs the contracts around the validated FX numerical backend. It does
**not** establish a Crab detection or calibrate the instrument. The user confirmed
Crab, uncompensated raw voltages, and Beijing time for the acquisition records.
The older `results/9178_reprocessed_20261015` images used a zenith-centered model;
they are engineering diagnostics, not Crab-centered sky maps.

## Implemented

- `packet9178.audit_file/audit_group` validate every selected datatype-13 packet,
  complete file length, modulo-2^32 counter continuity, and shared terminal counter
  range. The converter audits all required terminal files before channelization.
  Equal counters do not prove synchronized sampling clocks or local oscillators.
- Converter `--tstart` now means confirmed Unix UTC of sample zero in each raw file.
  Output epochs include both `--start-packet` and `--start-seq` offsets. Unknown zero
  remains zero. The NPZ includes the packet audit. The 250 MS/s and 1355 MHz defaults
  remain unconfirmed for observation 9178.
- `sky_geometry.SkyModel` builds an ICRS SIN tangent grid around the specified
  source, transforms it at each window midpoint to local ENU, and records UTC,
  source AltAz, Earth-orientation table provenance, and approximation bounds.
  AltAz Cartesian North/East/Up is explicitly reordered to East/North/Up.
- `run_lband_fx.py --sky-config` applies positive raw-voltage tracking gains and
  refreshes directions, center and gains each window. Both GPUs can process
  independent channel tiles. The existing explicit-geometry mode is preserved.
- Missing/ambiguous epoch, wrong frame, below-horizon source/grid, undersampled
  pixels and excessive within-window phase drift fail explicitly. IERS downloads
  are disabled during imaging; a local `iers_file` can be supplied when Astropy's
  bundled table does not cover the epoch. Data outside table coverage are rejected.
  Table status codes distinguish observations from predictions.

## Sky-config contract

The geometry NPZ must contain `positions_m[N,3]`, `frequencies_hz[C]`, and optional
voltage-multiply `gains[C,N,2]`, in exactly the input antenna/channel order. In sky
mode it needs no saved `directions` or `phase_center`. Positions must already be a
surveyed local ENU frame at the specified site. Merely subtracting a projected-map
origin does not establish the correct ENU orientation.

A **synthetic example**, not the 9178 acquisition epoch:

```json
{
  "input_start": "2018-01-11T22:00:00.000000000",
  "timezone": "Asia/Shanghai",
  "time_reference": "first_voltage_sample",
  "sequence_seconds": 0.00004,
  "source_ra_deg": 83.63322,
  "source_dec_deg": 22.01446,
  "site_lon_deg": 117.493,
  "site_lat_deg": 36.656,
  "site_height_m": 125.0,
  "coordinate_frame": "local_ENU",
  "image_size": 64,
  "pixel_deg": 0.005,
  "max_phase_drift_rad": 0.05
}
```

For recent epochs add `iers_file` pointing to an IERS-A `finals2000A.all` file;
relative paths resolve against the JSON directory. Preserve the file alongside
products for reproducibility. The runner records its SHA-256. Obtain it explicitly
from the [USNO IERS service](https://maia.usno.navy.mil/ser7/finals2000A.all).
See [Astropy IERS documentation](https://docs.astropy.org/en/stable/utils/iers.html)
and [AltAz conventions](https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html).

```bash
python scripts/audit_9178.py /path/to/9178/*4453.data > packet_audit.json
python scripts/run_lband_fx.py --input input.npz --geometry geometry.npz \
  --sky-config confirmed_sky.json --output new_output --devices 0,1 \
  --channel-tile 16 --window-sequences 1000 --coordinate-frame local_ENU \
  --iq-convention I+iQ --voltage-convention raw --remove-autocorr
```

`input_start` is the first voltage sample represented in **this NPZ**, including any
file/chunk offset, not the start of the recording session. `pixel_deg` is the SIN
projection scale; x points toward celestial east, y toward north. Images remain
uncalibrated instrumental products unless suitable measured gains are supplied.
Within each window geometry is constant at the midpoint. A conservative phase-error
bound uses the longest baseline and highest frequency; the default 0.05 rad limit
rejects overly long windows. For 480 m and 1355 MHz, 40 ms gives about 0.0201 rad.
The conservative spatial check uses the full baseline norm, including inactive
stands; this may reject an unnecessarily coarse grid but cannot silently certify it.

## Observation 9178 evidence and remaining blockers

All six copied files have 8184 complete packets, counter range
36443352..36451535, with no skipped, duplicated or reversed counters. The first
counter equals 4453 × 8184, consistent with the file suffix `_4453`.

Header words 60/61/62 admit a **candidate**, unconfirmed BCD date/time plus 10 ns
counter interpretation. H0 then reads `2026-01-11 14:29:27.879504820` initially.
The six first-packet subsecond values span about 0.975 ms despite identical packet
counters. Do not use those differences as sample offsets without the protocol.
If the header were UTC, it would correspond to Beijing 22:29:27 and Crab altitude
about 75.28 degrees at the illustrative site; if it were Beijing time, Crab would
be below the horizon. This supports investigating a UTC hardware clock but does
not establish the clock's timezone, receive/sample meaning, or precise sample epoch.

The accessible SSD, HDD, `/mnt/DATA` and `/mnt/quest_obs/data1` acquisition paths
were searched. Other recorder JSON files were found, including 9197, but no matching
9178 JSON was found. `/mnt/251`, `/mnt/share251`, and the old zjl04–06 mounts were
unavailable or timed out; parts of quest_obs were not readable. Other observation
JSONs cannot substitute for 9178 settings or timestamps.

`antenna_coordinates.csv` also contains conflicting coordinate representations:
converting its geodetic longitude/latitude/height to WGS84 geocentric coordinates
and comparing with its ITRF columns gives about **78.75 m for 东5**, **44.71 m for
东3**, and roughly **33 m for 东7/东8/东10**. Several north antennas agree below a
millimetre. This identifies a coordinate consistency problem, not which column
is authoritative. Do not silently choose one or label map X/E,Y/N columns as a
verified ENU survey. Original survey CRS/units/orientation and antenna/channel
mapping must be verified before a science-grade sky map.

Remaining work: recover the 9178 acquisition record and hardware time semantics;
verify sample rate/RF/channel sign/shared clock; reconcile the antenna survey;
estimate instrument delays and gains from suitable calibration data; then rerun
raw channelization and sky tracking. A sparse array's dirty image will still have
strong sidelobes and an extended Crab need not appear as a single point source.

## Validation and performance scope

The staged packet and sky tests were first observed failing, then passed after
implementation. The full suite has 64 passing tests and includes actual CUDA, dual-GPU tiling, non-zenith
raw point-source recovery, changing window geometry, tail windows, and converter
output epoch offsets. Synthetic four-antenna coherent input recovers XX/YY/ReXY=12
at the center after physical autocorrelations are removed, as N²−N predicts.
A two-sequence real-data rechannelization matches the previous input exactly;
unknown epoch remains zero. A durable synthetic two-GPU, three-window run is saved in `synthetic_sky/`;
all six channel/window chunks peak at the expected center. Logs and evidence are saved locally under
`/mnt/data_ssd/zdj/lwa_epic_crab_repair_20261016`.

This is a correctness stage, not a new throughput claim. Sky mode currently rebuilds
the FX context per window/tile to avoid stale steering. Reusing device allocations
and updating only steering/gains is a subsequent optimization to benchmark after
physical metadata is resolved. NPZ still decompresses the full input into CPU RAM;
long observations require the planned streaming input path.
