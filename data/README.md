# Dataset Access

This project uses the CICIDS2018 dataset as the primary benchmark for intrusion detection experiments. The dataset is publicly available and can be obtained from the official source:
https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv

Due to licensing restrictions, the dataset is not included in this repository and must be downloaded separately.
## Additional Dataset

For generalization analysis, the UNSW-NB15 dataset is also used. This dataset is not included in the repository and must be downloaded from official sources or Kaggle. After downloading, place the file in this directory as:

UNSW_NB15.csv
## Usage

The experiments use flow-level features derived from the datasets. Feature normalization and preprocessing follow the procedure described in the paper. No additional preprocessing beyond what is specified in the implementation is required. The UNSW dataset is automatically handled by the provided loader (unsw_loader.py).
## Notes

CICIDS2018 is used for primary evaluation (robustness analysis).
UNSW-NB15 is used for additional validation (generalization study).
Raw datasets are not redistributed due to licensing constraints.
