# BLOC-SHIELD
This repository provides the reference implementation of BLOC-SHIELD, a trust-aware and blockchain-assisted federated learning framework for intrusion detection in Software-Defined Networks (SDN).
The implementation corresponds to the experimental evaluation presented in the associated research paper. The framework is designed to improve robustness against poisoning attacks while preserving data locality across distributed SDN controllers, along with providing auditability through a lightweight blockchain layer.
## Experimental Scope
The primary experiments are conducted using:
* CICIDS2018 dataset
* Five SDN controllers as federated clients
* Non-IID data distribution across clients
* CNN-based intrusion detection model
* Trust-weighted federated aggregation
* Label flipping and model poisoning attacks
These experiments correspond directly to the main evaluation presented in the paper.
## Additional Evaluation (Generalization Study)

To assess generalizability, additional experiments are conducted using the UNSW-NB15 dataset.
* Dataset: UNSW-NB15
* Attack scenario: Label flipping
* Purpose: Validate performance consistency across different traffic distributions

To run this experiment:

1. Download the UNSW-NB15 dataset from official sources or Kaggle
2. Place the CSV file as:

data/UNSW_NB15.csv

3. Run the experiment:

python experiments/run_label_flip.py
## Repository Structure

* data/
  Dataset loaders and preprocessing scripts

* model/
  CNN-based intrusion detection model

* federated/
  Federated learning logic, including trust-based aggregation

* experiments/
  Scripts corresponding to experimental settings reported in the paper
## Reproducibility

All hyperparameters, training configurations, and evaluation protocols are fully specified in the research paper.

This repository provides:

* Reference implementation of the proposed framework
* Experimental scripts for key evaluation scenarios
* Dataset integration support
## Dataset Availability

* The CICIDS2018 dataset is not redistributed and must be obtained separately
* The UNSW-NB15 dataset must also be downloaded externally
## Notes

* The CICIDS2018 dataset is used for primary evaluation (robustness analysis)
* The UNSW-NB15 dataset is used for additional validation (generalization study)
* The blockchain component is used for auditability and does not directly influence model accuracy
## Citation

If you use this code, please cite the associated research paper.

(To be added after publication)
