# UCNF-FL

Implementation of **Reliability-Calibrated Neuro-Fuzzy Federated Learning for Heterogeneous Bearing Fault Diagnosis**.

## Installation

Requires Python >=3.11.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

Obtain original [CWRU bearing recordings](https://engineering.case.edu/bearingdatacenter/download-data-file), Paderborn or KAIST data. Use a consistent healthy/inner/outer label mapping. The recording CSV has `path,key,label,client,rig,group,split,fs,start,stop`; paths are relative to the CSV and MAT keys select one signal vector. Use `train,val,test`, actual client/rig identities and original sample rates. Complete bearings/files stay together. Deliberate chronological blocks need disjoint source intervals and at least 2048 target-rate samples of guard space.

```bash
ucnf-fl prepare --manifest data/recordings.csv --output data/bearings.npz
```

## Training and inference

```bash
ucnf-fl train --data data/bearings.npz --output runs/full --rounds 150 --local-epochs 5 --seed 17
```

## Evaluation

```bash
ucnf-fl evaluate --data data/bearings.npz --model runs/full/model.pt --output runs/full/evaluation --seed 17
ucnf-fl evaluate --data data/bearings.npz --model runs/full/model.pt --output runs/full/snr0 --snr 0 --seed 17
```

## Ablations

Variants: `full`, `fedavg`, `fedprox`, `no_reliability`, `no_class_support`, `no_brier`, `no_rule_regularization`, `no_quantization`. Use identical prepared clients, rounds and seeds.

```bash
ucnf-fl train --data data/bearings.npz --output runs/no_reliability --variant no_reliability --rounds 150 --local-epochs 5 --seed 17
ucnf-fl evaluate --data data/bearings.npz --model runs/no_reliability/model.pt --output runs/no_reliability/evaluation --seed 17
```

## Outputs

`runs/full/model.pt` stores the final-round model and payload accounting; `rounds.json` records client reliability, participation and communication. Evaluation writes `predictions.npz`, `metrics.json` and `clients.png`.
