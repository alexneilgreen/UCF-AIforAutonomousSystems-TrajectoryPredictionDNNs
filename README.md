<!-- SHOWCASE: false -->

# Trajectory Prediction Using DNNs

> A modified TUTR transformer model that predicts vehicle trajectories on the NuScenes dataset using single-mode and MinADE multi-mode loss approaches.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Spring%202025-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Special Topics: Artificial Intelligence for Autonomous Systems                                                                                                                                                                                                                                                                                                                                                   |
| Course Number          | EEL6938                                                                                                                                                                                                                                                                                                                                                                                                          |
| Semester               | Spring 2025                                                                                                                                                                                                                                                                                                                                                                                                      |
| Assignment Title       | Trajectory Prediction Using DNNs                                                                                                                                                                                                                                                                                                                                                                                 |
| Assignment Description | Implement and evaluate a modified version of the TUTR (Trajectory Unified Transformer) model on the NuScenes prediction challenge dataset. Part 1 trains a single-trajectory prediction model and reports ADE/FDE metrics. Part 2 implements a custom MinADE loss function that trains the model to generate 10 candidate trajectories and penalizes only the best-matching one, enabling multimodal prediction. |

---

## Project Description

This project adapts the TUTR (Trajectory Unified Transformer for Pedestrian Trajectory Prediction) architecture to the NuScenes autonomous driving dataset, predicting a target vehicle's position over 4 seconds (8 time indices) from 2 seconds (5 time indices) of observed state history for up to 10 neighboring agents. Two training regimes are evaluated: a baseline single-trajectory model using Smooth L1 loss, and a multi-trajectory model trained with a custom MinADE loss that selects the closest of 10 predicted paths to the ground truth. The MinADE approach yields substantially lower ADE and FDE, demonstrating the advantage of multimodal prediction in uncertain driving environments. The codebase was also adapted for Windows compatibility by wrapping the main execution block in an `if __name__ == "__main__":` guard.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

When run correctly, the script prints per-epoch loss and ADE/FDE metrics to the terminal, writes a per-epoch training log CSV, and saves a best-results CSV and model checkpoint to the `results/` and `checkpoint_*/` directories respectively. An ECDF plot of ADE and FDE distributions is also displayed at the end of training.

**Part 1 - Single Trajectory (baseline):**

```
Best Average minADE:          1.8851
Best Average minFDE:          4.1531
Best median minADE:           1.5393
Best median minFDE:           3.3809
Best 10th percentile minADE:  0.4601
Best 10th percentile minFDE:  0.7929
Best 90th percentile minADE:  3.7669
Best 90th percentile minFDE:  8.5885
```

**Part 2 - Multi-Trajectory with MinADE Loss:**

```
Best Average minADE:          0.8554
Best Average minFDE:          1.6460
Best median minADE:           0.6538
Best median minFDE:           1.0934
Best 10th percentile minADE:  0.3076
Best 10th percentile minFDE:  0.3467
Best 90th percentile minADE:  1.6143
Best 90th percentile minFDE:  3.4549
```

Lower ADE and FDE values indicate better prediction accuracy. The MinADE loss reduces average ADE by roughly 55% compared to the baseline. If results look unexpectedly high, verify that CUDA is available and that the dataset path (`./dataset/Nuscenes_data/`) is correctly populated. Adjusting `--n_clusters` or `--epoch` can also affect convergence.

---

## Key Concepts

`Trajectory Prediction` `Transformer Architecture` `MinADE Loss` `Multimodal Prediction` `Average Displacement Error` `NuScenes Dataset` `K-Means Motion Modes` `Autonomous Driving`

---

## Languages & Tools

- **Language:** Python 3.9
- **Framework/SDK:** PyTorch, TUTR (Trajectory Unified Transformer)
- **Hardware:** CUDA-compatible GPU (required for training)
- **Build System:** pip / Conda

---

## File Structure

```
project-root/
├── train_eval.py               # Baseline script (Part 1, minADEloss=False)
├── train_eval_modified.py      # Intermediate script with MinADE stub
├── train_eval_final.py         # Final script (Part 2, minADEloss=True by default)
├── requirements.txt            # Python dependencies
├── dataset/
│   └── Nuscenes_data/
│       ├── Train_Val_Sets.npz  # Train/validation split indices
│       ├── train/              # Per-sample .npz trajectory files (training)
│       └── test/               # Per-sample .npz trajectory files (evaluation)
├── TUTR_modified/
│   ├── transformer_encoder.py  # Transformer encoder module
│   ├── transformer_decoder.py  # Transformer decoder module
│   ├── model3.py               # TrajectoryModel4 definition
│   └── utils2.py               # Motion mode clustering utilities
├── checkpoint_*/               # Saved model weights (auto-created during training)
└── results/                    # Training logs and best-result CSVs (auto-created)
```

---

## Installation & Usage

### Prerequisites

- Python 3.9
- Conda (recommended)
- CUDA-compatible GPU with drivers installed
- NuScenes dataset (pre-processed, provided in project folder)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/alexneilgreen/UCF-AIforAutonomousSystems-TrajectoryPredictionDNNs.git
cd UCF-AIforAutonomousSystems-TrajectoryPredictionDNNs

# 2. Create and activate a Conda environment
conda create -n courseproject python=3.9
conda activate courseproject

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run Part 1 - single trajectory baseline
python train_eval.py

# 4b. Run Part 2 - multi-trajectory with MinADE loss
python train_eval_final.py --minADEloss True

# Optional: override hyperparameters
python train_eval_final.py --minADEloss True --n_clusters 75 --epoch 100 --lr 0.00005
```

### Controls

| Argument       | Default   | Description                               |
| -------------- | --------- | ----------------------------------------- |
| `--minADEloss` | `False`   | Enable MinADE multi-trajectory loss       |
| `--n_clusters` | `50`      | Number of motion mode clusters            |
| `--epoch`      | `50`      | Number of training epochs                 |
| `--lr`         | `0.00005` | Learning rate                             |
| `--num_k`      | `10`      | Number of predicted trajectory candidates |
| `--gpu`        | `0`       | GPU device index                          |

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.
