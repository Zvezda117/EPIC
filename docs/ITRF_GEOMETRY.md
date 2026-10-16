# Authoritative ITRF geometry — 2026-10-16

The user selected `ITRF_X/Y/Z` as the authoritative antenna coordinates. This
resolves the coordinate-source choice raised in the Crab observation audit.
Runtime geometry and station location now derive exclusively from those metre
columns. The original CSV is retained unchanged for traceability; its conflicting
legacy longitude/latitude and projected X/E,Y/N,Z/U columns are not used.

## One frame for all antenna selections

`src/python/array_geometry.py` uses the geocentric centroid of **all physical CSV
rows** as reference. It does not depend on the active antenna mapping or padding.
For the present 16-antenna survey this reference is:

- ITRF XYZ: (-2364952.832550977, 4544401.37112986, 3786945.369814889) m.
- WGS84 longitude: 117.49292883941544 degrees east.
- WGS84 geodetic latitude: 36.65640588472367 degrees north.
- Ellipsoidal height: 122.16020072894852 m.

For geodetic longitude λ and latitude φ, the Cartesian difference from that
reference is rotated with rows:

```text
East  = [-sinλ,         cosλ,        0]
North = [-sinφ cosλ,   -sinφ sinλ,   cosφ]
Up    = [ cosφ cosλ,    cosφ sinλ,   sinφ]
```

All three dimensions, including height, are retained. PAD inputs are placed at
ENU zero and must have zero voltages. Missing/duplicate names, invalid mappings,
nonfinite coordinates and coordinates inconsistent with terrestrial geocentric
metres fail explicitly. Changing survey rows changes the reference; exported
products record the reference, rotation, selected antenna order and CSV SHA-256.

The custom array path in `epic_utils.py`, the position readers in
`process_crab_data.py` and `process_9197_single_channel.py`, and the station-location
calculation in `convert_9197_correct.py` use this shared authority. The 9197
single-channel reader now uses the same configured ordering for voltages and
positions. These legacy scripts' other sampling/calibration approximations are
not validated by this geometry change; the sky-tracking FX runner remains the
explicit astronomical imaging path.

## Export geometry for FX

```bash
python scripts/build_itrf_geometry.py \
  --coordinates antenna_coordinates.csv \
  --mapping antenna_mapping_9178_full16.yaml \
  --frequency-start-hz 1353400000 --channel-width-hz 25000 --channels 128 \
  --output new_itrf_geometry.npz
```

Those frequency values reproduce the **existing assumed** 9178 channel selection;
the geometry repair does not confirm acquisition RF/sample-rate settings. The CLI
requires explicit first-channel RF center and signed increment, and refuses to
overwrite an existing file. Geometry NPZ fields are `positions_m`,
`frequencies_hz`, `antenna_names`, and `geometry_metadata_json`.

Use this NPZ with `run_lband_fx.py --sky-config ...`. Omit the three `site_*` fields
from the sky JSON to use the derived ENU reference automatically. If they are
provided, they must match the reference (1e-9 degrees, 1 mm tolerance); a conflicting
site fails before CUDA execution/output creation. Existing generic geometry files
without ITRF provenance retain their explicit-site contract. The sky config still
requires the confirmed first-sample epoch, timezone, source and integration/grid
parameters described in [Crab observation repair](CRAB_OBSERVATION_REPAIR.md).

## Validation

The 10 physical inputs of the full16 9178 mapping form 45 baselines. ITRF-to-ENU
baseline lengths agree within 5.69e-14 m; longest baseline is 477.172735754 m.
The old X/E,Y/N,Z/U baseline lengths differ from authoritative ITRF by at most
0.498882 m over these inputs. This is distinct from the previously reported
78.75 m conflict between the geodetic and ITRF representations of 东5.

Tests cover known equatorial axis directions and delay dot products, actual-array
baseline invariance, padding/reference stability, legacy readers, swapped antenna
indices, malformed inputs and incompatible sky-site rejection. A two-GPU test
injects raw point-source voltages using the original ITRF baselines and recovers
the source at the center through the exported ENU geometry. Ten coherent antennas
with physical autocorrelations removed give XX/YY/ReXY = 90, ImXY = 0.

Artifacts: `/mnt/data_ssd/zdj/lwa_epic_itrf_repair_20261016/` contains
`9178_itrf_geometry.npz`, `9178_enu_positions.csv`, `geometry.json`,
`baseline_validation.json`, and regression logs. No real Crab sky map is claimed:
the 9178 first-sample epoch and hardware time semantics remain unresolved.
