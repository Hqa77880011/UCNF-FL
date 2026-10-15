# UCNF-FL

Implementation of **Reliability-Calibrated Neuro-Fuzzy Federated Learning for Heterogeneous Bearing Fault Diagnosis**.

UCNF-FL combines the paper's five-block vibration encoder, 32-to-8 feature projection, twelve Gaussian TSK rules, learned probability temperature, and distinct aggregation paths for shared parameters and class consequents. Client reliability uses validation fuzzy entropy and Brier score. Absent classes cannot update their consequent slice. Eight-bit stochastic rounding uses persistent error feedback in both communication directions.

## Install and prepare

```bash
python -m venv .venv
python -m pip install -e .
```

Python 3.11+ and CPU PyTorch are sufficient for this single-process federated simulation. Obtain CWRU, Paderborn, and KAIST bearing recordings from their original maintainers. Use healthy/inner-race/outer-race labels 0/1/2 consistently. The manuscript selects CWRU drive-end at 0.014-inch defects and 6 o'clock outer-race position; Paderborn conditions N15_M07_F10, N09_M07_F10, N15_M01_F10, N15_M07_F04; and KAIST housing-A x direction at 3010 rpm under 0/2/4 Nm. These produce 4/4/3 physical-condition clients.

Create `recordings.csv` with `path,key,label,client,rig,group,split,fs,start,stop`. `path` is relative to the CSV. MAT `key` selects a one-channel vector; NPY inputs are a vector. `fs` is the original sample rate, and optional `start`/`stop` select a half-open acquisition interval. Use `train`, `val`, `test` splits and group entire bearings/files together. If a CWRU file is deliberately split into chronological blocks, use distinct block group IDs and at least 2048 target-rate samples of guard space between differently partitioned intervals. The preparer checks source-interval overlap/guards.

```bash
ucnf-fl prepare --manifest recordings.csv --output bearings.npz
ucnf-fl train --data bearings.npz --output outputs/full --rounds 150 --local-epochs 5 --seed 17
ucnf-fl evaluate --data bearings.npz --model outputs/full/model.pt --output outputs/full_eval
```

Preparation removes each selected interval's mean, applies a fourth-order 10 Hz zero-phase high-pass filter, polyphase-resamples to 12.8 kHz, segments 2048-point windows at stride 1024, and standardizes with each client's training mean/std. Filters never cross the supplied acquisition interval. Training adds circular shifts up to 64 samples and amplitude factors 0.90/0.95/1/1.05/1.10. Evaluation uses no augmentation unless SNR stress is requested.

The local objective is inverse-square-root-class-weighted focal loss (gamma two), 0.20 multiclass Brier, and 0.01 rule-utilization loss. AdamW uses weight decay 1e-4 and a round-wise cosine learning rate from 0.001 to 1e-5. Local validation provides entropy/Brier and class-specific summaries; when an observed training class has no local validation sample, its class reliability uses the client's overall summaries. Reliability min–max normalization is across participating clients, with floor 0.2, entropy coefficient one, and Brier coefficient 1.5. Constant summaries map to zero. Class slices with no participating support are retained exactly, including preservation of pending downlink quantization error until support returns.

## Baselines and component controls

```bash
ucnf-fl train --data bearings.npz --output outputs/fedavg --variant fedavg
ucnf-fl train --data bearings.npz --output outputs/fedprox --variant fedprox
ucnf-fl train --data bearings.npz --output outputs/no_reliability --variant no_reliability
ucnf-fl train --data bearings.npz --output outputs/no_class_support --variant no_class_support
ucnf-fl train --data bearings.npz --output outputs/no_brier --variant no_brier
ucnf-fl train --data bearings.npz --output outputs/no_rule_regularization --variant no_rule_regularization
ucnf-fl train --data bearings.npz --output outputs/float32 --variant no_quantization
```

FedAvg uses sample-count model averaging with the same fuzzy architecture and local objective; FedProx additionally uses coefficient 0.01 in the squared-distance proximal objective. Both send float32 parameters. Other variants remove only the named full-method mechanism. `--rules 6|12|18` controls rule count. Fixed final-round checkpoints avoid test-dependent selection.

`--participation 7` selects seven clients for local computation per round. Every logical client receives the common downlink in this simulator, including nonparticipating clients; it therefore measures computation participation rather than a network-offline protocol. Downlink accounting includes every such recipient. Real disconnected deployments need a declared synchronization transport and its extra payload.

Eight-bit payload counts one byte per scalar plus a four-byte scale per tensor. Shared and class-aware weights aggregate quantized uplink deltas; a quantized server delta updates the shared global model. Persistent residuals hold unsent quantization error. Float32 controls count four bytes per scalar. Validation summary and class-count bytes are recorded separately from parameter payload.

## Evaluate noise, heterogeneity, and outputs

```bash
ucnf-fl evaluate --data bearings.npz --model outputs/full/model.pt --output outputs/snr0 --snr 0 --seed 17
ucnf-fl evaluate --data bearings.npz --model outputs/full/model.pt --output outputs/snr10 --snr 10 --seed 17
```

Noise power is scaled from each clean window's power. Label-skew, missing-class, rare-prevalence, and label-corruption studies use separate manifests/prepared datasets with only training assignments/labels changed; retain the same held-out groups and seed for paired comparisons. Do not remove test classes to manufacture a missing-training-class advantage.

`rounds.json` records participation, client reliability, shared/class weights, and cumulative uplink/downlink/summary bytes. `model.pt` stores only model parameters and payload accounting. `predictions.npz` retains probabilities, true labels, client, rig, and acquisition group for statistical aggregation. `metrics.json` includes accuracy, balanced accuracy, Macro-F1, 15-bin confidence ECE, Brier score, equal-rig Macro-F1, worst-client Macro-F1, client population standard deviation, parameter count, and communication. `clients.png` visualizes client disparities.

For the paper's aggregate protocol use seeds 17/29/43/71/101 and the registered five-fold bearing/block assignments. Compute seed means and sample standard deviations from independently run outputs; model performance is not prefilled. The code implements the federated updates directly and does not require access to a remote training service.
