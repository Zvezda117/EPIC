#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/bin/activate epic310

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_SEQS=53634
CHUNK_SEQS=10000
CHANNELS=128
IMAGERES=0.02
CNIB_SCALE=0.0156822

for ((start=0; start<TOTAL_SEQS; start+=CHUNK_SEQS)); do
  remain=$((TOTAL_SEQS - start))
  if (( remain > CHUNK_SEQS )); then
    seqs=$CHUNK_SEQS
  else
    seqs=$remain
  fi

  ms_float="$(python -c "print(${seqs} * 0.04)")"
  ms_int="$(python -c "print(int(round(${seqs} * 0.04)))")"
  tag="$(printf "%05d" "$start")"
  npz="data/epic_ready/9178_crab_full16_seq${tag}_${seqs}seq_128chan_fixedscale.npz"
  out="${ROOT_DIR}/results/9178_crab_full16_seq${tag}_${seqs}seq_fixedscale_res002"

  echo "=== 9178 chunk start=${start} seqs=${seqs} ms=${ms_float} ==="
  python scripts/convert_9178_to_epic.py \
    --array-mode full16 \
    --output "$npz" \
    --channels "$CHANNELS" \
    --integration-ms "$ms_float" \
    --start-seq "$start" \
    --iq-order IQ \
    --cnib-scale "$CNIB_SCALE" \
    --no-compress

  PYTHONPATH="${ROOT_DIR}/src/python" \
  EPIC_ARRAY_CONFIG="${ROOT_DIR}/antenna_mapping_9178_full16.yaml" \
  EPIC_ANTENNA_COORDS="${ROOT_DIR}/antenna_coordinates.csv" \
  EPIC_CENTER_FREQ_MHZ=1355 \
  EPIC_GULP_NPZ="${ROOT_DIR}/${npz}" \
  ./build/epic++ --offline \
    --out_dir "$out" \
    --disable_metrics \
    --imagesize 64 \
    --channels "$CHANNELS" \
    --chan_nbin 4 \
    --seq_accum "$ms_int" \
    --nts "$seqs" \
    --imageres "$IMAGERES" \
    --nstreams 1 \
    --support 5 \
    --aeff 25 \
    --kernel_oversample 2 \
    --ngpus 1 \
    --gpu_ids=0
done

python scripts/combine_9178_fits.py
