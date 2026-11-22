#!/bin/sh
# Unblur bake-off, second stage (UNBLUR.md): the shortlist from the screening run
# (results/quality_unblur_screen*.csv) on the full real sets and 9 photos per synthetic set,
# then the tile-context study, dtype check, Kohler renders and latency. Sequential on purpose,
# so timings are not skewed by other GPU jobs.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
SHORT="fftformer-realblurj stripformer-realblurj ufpnet-realblurj restormer-motion \
       lakdnet-defocus-lfdpdd-s drbnet-lfdpdd lakdnet-defocus-dpdd-l restormer-defocus"
Q="$PY quality.py --op unblur --overlap 64 --csv quality_unblur.csv"

# 1. full real sets (+ the input row), 9 photos per synthetic set and the sharp set
$Q --models input --sets unblur/realblurj unblur/dpdd unblur/realdof
$Q --models $SHORT --sets unblur/realblurj unblur/dpdd unblur/realdof
$Q --models input --sets unblur/shake15-25 unblur/shake30-45 unblur/defocus4 unblur/defocus8 unblur/defocus12 unblur/sharp --limit 9
$Q --models $SHORT --sets unblur/shake15-25 unblur/shake30-45 unblur/defocus4 unblur/defocus8 unblur/defocus12 unblur/sharp --limit 9
# EVSSM (~9x slower than Stripformer) was only screened (quality_unblur_screen2.csv); its full
# RealBlur-J run was skipped because speed alone rules it out (UNBLUR.md).

# 2. context: does deblurring need more than a 512 px tile? (tile 1024/128 and whole image)
C="$PY quality.py --op unblur --csv quality_unblur_context.csv --limit 4 --no-save \
   --sets unblur/shake30-45 unblur/defocus12 unblur/realdof unblur/realblurj"
$C --models fftformer-realblurj stripformer-realblurj lakdnet-defocus-lfdpdd-s drbnet-lfdpdd restormer-defocus --overlap 64 --tile 512 --tag @t512
$C --models fftformer-realblurj stripformer-realblurj lakdnet-defocus-lfdpdd-s drbnet-lfdpdd restormer-defocus --overlap 128 --tile 1024 --tag @t1024
$C --models fftformer-realblurj stripformer-realblurj lakdnet-defocus-lfdpdd-s drbnet-lfdpdd restormer-defocus --tile 0 --tag @whole

# 3. precision
$PY dtype_check.py --models $SHORT nafnet-gopro-w64 evssm-realblurj --sets realblurj dpdd shake30-45 sharp --n 2

# 4. real camera shake without GT, for the contact sheets
$PY quality.py --op unblur --overlap 64 --render --sets unblur/kohler \
    --models fftformer-realblurj stripformer-realblurj ufpnet-realblurj evssm-realblurj lakdnet-defocus-lfdpdd-s restormer-motion

# 5. latency (alone on the GPU): auto dtype, then the picks in fp16
$PY latency.py --op unblur --overlap 64 --csv latency_unblur.csv \
    --models fftformer-realblurj stripformer-realblurj ufpnet-realblurj lakdnet-defocus-lfdpdd-s drbnet-lfdpdd restormer-defocus
$PY latency.py --op unblur --overlap 64 --csv latency_unblur.csv --dtype fp16 \
    --models stripformer-realblurj lakdnet-defocus-lfdpdd-s drbnet-lfdpdd
echo UNBLUR STAGE2 DONE
