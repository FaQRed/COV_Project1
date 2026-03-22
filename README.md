# 🐾 Oxford-IIIT Pet Classification

Classification of 37 cat and dog breeds using CNN (PyTorch).  
**Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 7,349 images, 37 breeds

---

## Experiments

| # | Model | Key Changes | Test Accuracy |
|---|-------|-------------|:-------------:|
| 0 | Baseline CNN | 3x Conv-ReLU-Pool, Adam | 13.3% |
| 1 | BatchNorm + Deeper | 4 blocks, BatchNorm after each Conv | 22.4% |
| 2 | Dropout + Augmentation | Dropout(0.5), flip/rotate/colorjitter | 22.5% |
| 3 | Exp1 + Aug + SGD | SGD momentum=0.9, augmentation | 29.0% |
| 4 | **Deep BN + GAP** | 5 blocks, BatchNorm, Global Average Pooling | **32.3%** |

---

## Learning Curves

| Baseline | Exp1 — BatchNorm                                                       |
|----------|------------------------------------------------------------------------|
| ![](plots/learning_curve_exp_0_Baseline_CNN.png) | ![](plots/learning_curve_exp_1_Exp1_%E2%80%94_BatchNorm_%2B_Deeper_network.png) |

| Exp2 — Dropout + Aug | Exp3 — SGD |
|----------------------|------------|
| ![](plots/learning_curve_exp_2_Exp2_%E2%80%94_Dropout_%2B_Augmentation.png) | ![](plots/learning_curve_exp_3_Exp3_%E2%80%94_Exp1_%2B_Augmentation_%2B_SGD.png) |

| Exp4 — Best model |
|---|
| ![](plots/learning_curve_exp_4_Exp4_%E2%80%94_Deep_BN_%2B_GAP_%2B_Dropout.png) |

---
 
## Best Model — Exp4 (32.3%)
 
**Easiest breeds:**

| Breed | F1-score |
|-------|:--------:|
| Egyptian Mau | 0.57 |
| Keeshond | 0.44 |
| Samoyed | 0.45 |
| Leonberger | 0.46 |
| Newfoundland | 0.46 |
 
**Hardest breeds:**

| Breed | F1-score |
|-------|:--------:|
| American Bulldog | 0.06 |
| American Pit Bull Terrier | 0.14 |
| Staffordshire Bull Terrier | 0.16 |
| Sphynx | 0.22 |
| Siamese | 0.23 |
 
---

## Conclusions

- **BatchNorm** stabilizes training and speeds up convergence (+9% vs baseline)
- **Global Average Pooling** reduces overfitting compared to Flatten
- **SGD with momentum** outperformed Adam when combined with augmentation
- Visually similar breeds (Bulldog / Pit Bull / Staffordshire) are the hardest to classify