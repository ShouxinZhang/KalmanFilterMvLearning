This repository hosts my research project on the Kalman filter. It includes Python 3 implementations and accompanying study notes in Markdown and TeX. 

This project was initiated based on the advice of my senior classmate.

## Reproduce the paper experiment (FDEKF / RR-FDEKF vs Adam)

The paper PDF and the measurement dataset are under `FDEKF/`.

- Dataset: `FDEKF/SISO.mat` (uses `txa` as input and `rxa_flt` as target by default)
- Script: `FDEKF/reproduce_adam_ekf.py`

Run (quick sanity run on a subset):

```bash
.venv/bin/python FDEKF/reproduce_adam_ekf.py --algo all --epochs 5 --limit 2000 --plot
```

Outputs are written to `FDEKF/out/` (JSON metrics + PNG curves + learned parameters).
