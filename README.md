# Disentangling Depression from Identity: A Domain-Adversarial Framework


This repository contains the official implementation of the paper: **"Disentangling Depression from Identity: A Domain-Adversarial Framework for Privacy-Preserving Personalized Speech Depression Detection."**



We propose a **Domain-Adversarial Neural Network (DANN)** to solve the "Identity Leakage" problem in personalized depression detection. By treating speaker identity as a nuisance variable, our model unlearns voiceprints while retaining diagnostic pathological cues.

---

## 🚀 Key Results (Highlights)

Our framework achieves **State-of-the-Art (SOTA)** performance while drastically reducing privacy risks.

| Method | Scenario | Depression F1 | Depression Acc | Speaker Acc (Privacy Leakage) $\downarrow$ |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | Personalized (History) | 0.785 | 79.0% | **91.9%** (Severe Leakage ⚠️) |
| **Augmentation** | Pitch Shift | 0.731 | 74.0% | 20.0% (Failed Disentanglement) |
| **Ours (DANN)** | **Adversarial** | **0.813** | **81.4%** | **3.2%** (96% Reduction 🛡️) |

> **Note:** The Speaker Accuracy of 3.2% approaches the random chance level ($\approx 0.53\%$), demonstrating effective anonymization.

---

## 📂 Repository Structure

```text
.
├── run_dann.py                # Main DANN training script (Includes 5-run validation & t-SNE)
├── run_aug_baseline.py        # Baseline: Data Augmentation via Pitch Shifting
├── run_speaker_probe_final.py # Privacy Probe: Measures identity leakage of the Standard model
├── run_linear_probing_hf.py   # Standard Baseline: Linear Probing on Wav2Vec2 features
├── experiment_generator.py    # Data Splitter: Generates size-matched train/test sets
├── plot_tsne.py               # Visualization: Generates Figure 2 (t-SNE plots)
├── plot_results2.py           # Visualization: Generates Confusion Matrix & Performance Bar Charts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
