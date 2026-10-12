# SafeAdaptiveSurveillance

Adaptive surveillance system for video anomaly detection using deep learning, safe continual learning, and human-in-the-loop verification.

## Overview

This project focuses on the challenge of adapting video anomaly detection models to real-world surveillance environments without blindly absorbing anomalous patterns into the learned normality model. The system combines:

- deep learning-based anomaly scoring
- uncertainty-driven candidate buffering
- human-in-the-loop verification
- replay-based adaptation
- validation-based rollback safeguards

The goal is to support legitimate scene drift while reducing anomaly contamination and unnecessary manual intervention.

## Research Motivation

Traditional static anomaly detection models often fail when camera environments change over time due to illumination shifts, crowd density variations, or scene layout changes. However, naive online adaptation can cause a model to learn persistent anomalous events as normal, leading to false negatives and unsafe behavior.

This project addresses that issue by applying a safe adaptation pipeline inspired by the principle: "Learn continuously, but not learn blindly."

## Key Features

- Convolutional autoencoder-based normality modeling
- Reconstruction error scoring for anomaly detection
- Threshold-based frame classification
- Candidate buffering for ambiguous observations
- Sparse human verification for safety-critical decisions
- Replay-based update protection
- Validation-driven rollback to retain safe model behavior
- UCSD Ped2 benchmark support

## Project Structure

```text
AdaptiveSurveillanceResearch/
├── src/
│   ├── buffer/
│   │   ├── candidate_buffer.py
│   │   └── safe_gate.py
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── download_ped2.py
│   │   ├── generate_demo_data.py
│   │   └── ped2_dataset.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   └── spatial_temporal_ae.py
│   ├── demo_hitl_verification.py
│   ├── evaluate.py
│   ├── main_pipeline.py
│   ├── train.py
│   └── __init__.py
├── tools/
│   └── generate_proposed_architecture.py
├── generate_anomaly_threshold_plot.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/munni036/AdaptiveSurveillanceSystem.git
cd AdaptiveSurveillanceSystem
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Demo human-in-the-loop verification

```bash
python src/demo_hitl_verification.py
```

### Main pipeline

```bash
python src/main_pipeline.py
```

### Training baseline model

```bash
python src/train.py
```

## Dataset

The project is designed for the UCSD Ped2 dataset. The dataset can be downloaded via the pipeline utilities in the data module. If the dataset is unavailable, the code includes a demo fallback for generating synthetic example frames.

## License

This project is intended for academic and research use.

## Citation

If you use this project in your research, please cite the repository and the associated paper work.

## Contact

For questions or collaboration inquiries, feel free to reach out through the GitHub repository or project maintainers.
