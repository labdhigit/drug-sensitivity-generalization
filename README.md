# Data (not committed: download from the sources below)

Place these three files in this folder:

| File | Columns | Source |
|---|---|---|
| expression.csv | rows = cell lines (index), columns = genes, log2 expression | GDSC expression or CCLE/DepMap (https://depmap.org/portal/download/) |
| drug_response.csv | cell_line, drug, ln_ic50 | GDSC bulk download (https://www.cancerrxgene.org/downloads/bulk_download) |
| drug_smiles.csv | drug, smiles | GDSC drug annotation file (same page) |

Cell-line IDs must be harmonised across files. Report the number of cell lines that survive the join.
