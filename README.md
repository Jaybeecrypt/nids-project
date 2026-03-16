# nids-project
Cloud-Based Hybrid NIDS Using SMOTE-XGBoost-CNN — Bowen University Thesis

Multi-class network intrusion detection on CICIDS-2017 using a three-stage
pipeline: SMOTE class balancing → XGBoost feature selection → 1D-CNN classification.

**[▶ Live Demo](https://YOUR-USERNAME.github.io/nids-project/)**

---

## Results

Evaluated on 756,240 held-out test samples from CICIDS-2017.

| Metric | Score |
|--------|-------|
| Accuracy | 93.06% |
| Weighted F1 | 94.45% |
| Weighted Precision | 96.60% |
| Weighted Recall | 93.06% |
| Training stopped | Epoch 21 / 50 |
| Best val accuracy | 93.63% |

---

## Pipeline
```
CICIDS-2017  →  Preprocessing  →  SMOTE  →  XGBoost  →  1D-CNN  →  Prediction
3.1M samples    StandardScaler    75k bal    78→35 feat   149k params  15 classes
```

**Why this order matters:**
XGBoost feature importance scores are unreliable on imbalanced data — SMOTE
runs first so all 15 classes are equally represented before feature ranking.
The CNN then trains only on the 35 most discriminative features, reducing
noise and training time without sacrificing accuracy.

---

## Dataset

CICIDS-2017 — Canadian Institute for Cybersecurity, University of New Brunswick.
Five days of labelled network traffic (July 3–7, 2017) across 15 classes.

| Property | Value |
|----------|-------|
| Raw samples | 3,119,345 |
| After cleaning | 2,520,798 |
| Features | 78 (CICFlowMeter) |
| Train / Test split | 70 / 30 stratified |
| Most common class | BENIGN — 75.62% |
| Rarest class | Heartbleed — 11 samples |

---

## SMOTE Configuration

Standard k=5 was not viable for classes with fewer than 6 real training
samples (Heartbleed: 8, SQL Injection: 15, Infiltration: 25). k=1 was
used across all classes with a cap of 5,000 samples per class.
```python
smote = SMOTE(k_neighbors=1, random_state=42)
# Result: 15 classes × 5,000 = 75,000 balanced training samples
```

---

## Model
```python
model = Sequential([
    Input(shape=(35, 1)),
    Conv1D(64,  kernel_size=3, activation='relu'),  # 256 params
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),  # 24,704 params
    MaxPooling1D(pool_size=2),
    Flatten(),                                       # → 896
    Dense(128, activation='relu'),                   # 114,816 params
    Dropout(0.5),
    Dense(64,  activation='relu'),                   # 8,256 params
    Dense(15,  activation='softmax'),                # 975 params
])
# Total: 149,007 params | ~600 KB
```

Compiled with `Adam(lr=0.001)` and `sparse_categorical_crossentropy`.
Trained for up to 50 epochs with `EarlyStopping(patience=5)`.

---

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| BENIGN | 1.00 | 0.92 | 0.96 | 628,518 |
| DDoS | 0.85 | 1.00 | 0.92 | 38,404 |
| DoS Hulk | 0.84 | 1.00 | 0.91 | 51,854 |
| PortScan | 0.72 | 1.00 | 0.84 | 27,208 |
| DoS GoldenEye | 0.74 | 0.99 | 0.85 | 3,086 |
| FTP-Patator | 0.70 | 0.99 | 0.82 | 1,779 |
| DoS Slowhttptest | 0.61 | 0.99 | 0.75 | 1,568 |
| DoS slowloris | 0.61 | 0.99 | 0.75 | 1,616 |
| SSH-Patator | 0.60 | 0.94 | 0.74 | 966 |
| Bot | 0.04 | 1.00 | 0.08 | 584 |
| Web Attack – XSS | 0.04 | 0.99 | 0.07 | 196 |
| Heartbleed | 0.07 | 1.00 | 0.13 | 3 |
| Web Attack – Brute Force | 0.13 | 0.13 | 0.13 | 441 |
| Infiltration | 0.00 | 1.00 | 0.01 | 11 |
| Web Attack – SQL Injection | 0.01 | 0.50 | 0.01 | 6 |

Low precision on Heartbleed, Infiltration, and SQL Injection is expected —
SMOTE synthesised these classes from 8–25 real samples, producing wide
decision boundaries. Recall of 1.00 means no instances were missed, which
is the correct priority for a security classifier.

---

## Stack

Python 3.9 · pandas · NumPy · imbalanced-learn · XGBoost · TensorFlow 2.x · scikit-learn

---

## Repository Structure
```
nids-project/
└── index.html    # Interactive implementation page (live demo)
```

---

*Olabode Ajibola · BU22CSC1118 · Bowen University, Iwo*
*Supervisor: Miss Busolami Oluwadamilare · December 2025*
