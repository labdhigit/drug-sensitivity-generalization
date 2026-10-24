# Predicting Anti-Cancer Drug Sensitivity from Baseline Gene Expression: Cell-Line- and Drug-Wise Generalization Testing

**Author:** Labdhi Kumar Jain (23BCB0147), Jeevan P Abraham (24BCB0083)
**Role:** 

Labdhi Kumar Jain: data and evaluation

Literature review (GDSC, CCLE, leakage in drug-response prediction)
Sourcing GDSC and CCLE/DepMap data and harmonising cell-line IDs
Building drug features (Morgan fingerprints from SMILES)
Designing the three evaluation regimes and nested cross-validation
Writing the Introduction, Methods and Data Availability sections

Jeevan P Abraham: modelling and interpretation

Implementing and tuning the models (ElasticNet/Ridge, Random Forest, XGBoost)
Running experiments across regimes and seeds, with confidence intervals
SHAP analysis and checking genes against known markers (ERBB2, EGFR)
Writing the Results and Discussion and Conclusion sections


## What this project does
Predicts ln(IC50) drug response of a cancer cell line from its baseline gene expression plus drug
fingerprints (Morgan/ECFP from SMILES), and measures how much reported accuracy is leakage by
comparing three evaluation regimes:

1. Random split (naive baseline, leaky)
2. Leave-cell-lines-out (new tumours)
3. Leave-drugs-out (new compounds; needs drug features)

Gene selection (variance filter) is fitted inside each training fold only.
Models: ElasticNet (baseline), XGBoost (main). Planned: Random Forest, SHAP, multi-task NN (stretch).

## Status
Code pipeline written and smoke-tested on synthetic data. Real-data experiments: in progress.
No real results are reported in this repository yet.

## Run
```
pip install -r requirements.txt
python src/pipeline.py --synthetic          # smoke test only, numbers are meaningless
python src/pipeline.py --data data          # real data, see data/README.md
```

## Data
GDSC (https://www.cancerrxgene.org/downloads/bulk_download),
CCLE via DepMap (https://depmap.org/portal/download/). See `data/README.md`.

## References
- Iorio et al., Cell (2016): GDSC
- Ghandi et al., Nature (2019): CCLE
- Lundberg & Lee, NeurIPS (2017): SHAP
