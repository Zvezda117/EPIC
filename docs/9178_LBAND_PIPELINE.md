# 9178 L-band Array Pipeline

9178 is not a standard LWA station dataset. In this checkout EPIC is used only
as the MOFF imager; antenna positions, wiring, and calibration must come from
the local L-band array.

## Wiring Used

The authoritative antenna positions are in `antenna_coordinates.csv`.

| Terminal | File prefix | ch0 | ch1 |
| --- | --- | --- | --- |
| H00 | HBaseband0Spec0 | 北4 V | 北3 V |
| H11 | HBaseband1Spec1 | 东4 H | 东3 V |
| H22 | HBaseband2Spec2 | 东5 H | 东6 H |
| V00 | VBaseband0Spec0 | 东4 V | 东5 V |
| V11 | VBaseband1Spec1 | 东9 H | 东10 V |
| V22 | VBaseband2Spec2 | 东7 H | 东8 V |

`antenna_mapping_9178_epic4.yaml` and `--array-mode epic4` are only a quick
four-input smoke test. The full 9178 wiring uses ten physical antennas. EPIC's
CUDA path is happiest with its original 32-input layout, so the full imaging
mode pads the ten physical antennas to sixteen stands with six zero-input
placeholders: use `antenna_mapping_9178_full16.yaml` and `--array-mode full16`.

## Packet Layout

Each 9178 file has 8184 packets. Each packet is:

- 256-byte common header, tag `0x18EFDC01`
- 262144-byte payload
- datatype 13, int8 IQ
- two terminal channels
- payload layout repeats `[32 bytes ch0][32 bytes ch1]`
- each terminal channel has 65536 complex samples per packet

Validate the fast deinterleaver against the verified `UnpackCustom` path:

```bash
source /opt/miniconda3/bin/activate epic310
cd /mnt/data_ssd/zdj/LWA_EPIC
python scripts/check_9178_interleave.py
```

## Imaging Inputs

EPIC expects channelized complex voltages, not raw baseband. The converter uses
40 us sequences:

- sample rate: 250 MS/s
- samples per EPIC sequence: 10000
- 25 kHz FFT-bin spacing

For a 40 ms image:

```bash
python scripts/convert_9178_to_epic.py \
  --array-mode epic4 \
  --output data/epic_ready/9178_crab_epic4_40ms_128chan.npz \
  --channels 128 \
  --integration-ms 40
```

Run the current four-stand EPIC binary:

```bash
PYTHONPATH=/mnt/data_ssd/zdj/LWA_EPIC/src/python \
EPIC_ARRAY_CONFIG=/mnt/data_ssd/zdj/LWA_EPIC/antenna_mapping_9178_epic4.yaml \
EPIC_ANTENNA_COORDS=/mnt/data_ssd/zdj/LWA_EPIC/antenna_coordinates.csv \
EPIC_CENTER_FREQ_MHZ=1355 \
EPIC_GULP_NPZ=/mnt/data_ssd/zdj/LWA_EPIC/data/epic_ready/9178_crab_epic4_40ms_128chan.npz \
./build/epic++ --offline \
  --out_dir /mnt/data_ssd/zdj/LWA_EPIC/results/9178_crab_epic4_40ms_res002 \
  --disable_metrics --imagesize 64 --channels 128 --chan_nbin 4 \
  --seq_accum 40 --nts 1000 --imageres 0.02 \
  --nstreams 1 --support 5 --aeff 25 --kernel_oversample 2 \
  --ngpus 1 --gpu_ids=0
```

`--imageres 0.02` is important for this geometry. With `0.05`, some selected
antennas land outside a 64x64 grid and the image can become zero or badly
clipped.

## Calibration Status

Current 9178 EPIC runs are not fully calibrated. For custom arrays,
`epic_utils.gen_phases_lwasv()` returns unity phases, so the current image uses:

- CSV antenna positions for gridding
- no cable delay calibration
- no per-input complex gain calibration
- no Crab geometric phase tracking

The existing 9197 scripts contain approximate CasA geometric corrections and a
point-source phase-slope self-calibration experiment. They are useful examples
but are not a valid 9178 Crab calibration as-is.

A practical calibration path for this L-band array is:

1. Confirm observation time and time scale from the acquisition system or packet
   header.
2. Use the CSV coordinates to compute Crab topocentric geometric delays for each
   input at that time.
3. Solve residual instrumental delays from a longer Crab interval by fitting
   baseline phase versus frequency.
4. Convert baseline delays to per-input delays with one reference input fixed.
5. Apply per-input delay phases before CNib quantization, then image.
6. Solve complex bandpass gains only after delay residuals are stable.

The quick 40 ms and 400 ms phase-slope checks currently have large residuals on
most baselines, so they should be treated as diagnostics, not final calibration.

## Useful Files

- `scripts/convert_9178_to_epic.py`: current 9178 converter
- `scripts/check_9178_interleave.py`: deinterleave validation
- `antenna_mapping_9178.yaml`: full ten-antenna wiring
- `antenna_mapping_9178_epic4.yaml`: current four-stand smoke-test mapping
- `antenna_coordinates.csv`: authoritative local-array coordinates
- `data/epic_ready/`: generated EPIC NPZ inputs
- `data/calibration/`: diagnostic phase-slope CSV outputs
- `results/9178_*`: generated FITS and quicklook PNG outputs
