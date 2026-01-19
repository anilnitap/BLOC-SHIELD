# BLOC-SHIELD

This repository provides the reference implementation of BLOC-SHIELD, a trust based and blockchain assisted federated learning framework for intrusion detection in software defined networks.

The implementation corresponds directly to the experimental evaluation presented in the associated research paper. The framework is designed to improve robustness of federated intrusion detection against poisoning attacks while preserving data locality across SDN controllers.

## Experimental Scope

The experiments are conducted using:
- CIC IDS 2018 dataset
- Five SDN controllers as federated clients
- Non IID data distribution
- CNN based intrusion detection model
- Trust weighted federated aggregation
- Label flipping and model poisoning attacks

No additional experiments beyond those reported in the paper are included.

## Repository Structure

- data: Dataset access instructions
- model: Intrusion detection model definition
- federated: Federated learning and trust aggregation logic
- experiments: Scripts corresponding to experimental settings in the paper


## Reproducibility

All hyperparameters, training settings, and evaluation protocols are fully specified in the paper. This repository provides code structure and reference implementation to support reproducibility.

The CIC IDS 2018 dataset is not redistributed and must be obtained separately.
