# NEXT_MEETING_SUMMARY

## 1) Current repo status
Pipeline updated for tactical imperfect-data scenario controls and experiment scripting.
## 2) Datasets supported
LANL and OPTC are supported. IoT is TODO.
## 3) Already implemented
TopoGCL and GNN baselines exist; corruption utilities were extended.
## 4) Tactical scenarios
- low_volume: less observed data
- missing_structure: missing nodes/edges
- interference: noisy/delayed/corrupted observations
## 5) Training pipeline
Load clean -> split benign -> apply train degradation -> train.
## 6) Inference pipeline
Apply test degradation only to holdout benign+malicious graphs -> evaluate.
## 7) Experiment matrix
clean->degraded, degraded->degraded, mild(0.2)->severe(0.4) across 0.0..0.5 sweeps.
## 8) Metrics template
Accuracy, Precision, Recall, F1, FPR, AUROC.
## 9) Result locations
results/*, figures/*, paper_tables/*
## 10) Remaining TODOs
IoT integration; run full sweeps when full dataset files are pulled via git-lfs.
