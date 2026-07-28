# Lab 03: Computer Vision with Classical Machine Learning

**Course:** ITAI 1378 — Houston Community College
**Module:** 03
**Student:** Alexia Y. Chavez
**Platform:** Google Colab / scikit-learn

---

## Problem Statement

Face recognition is a classic computer vision task: given a photo of a face, identify which of a known set of people it belongs to. This lab tackles it *without* deep learning, using the **Olivetti Faces** dataset (400 grayscale images across 40 subjects, 10 images each) and classical machine learning — feature engineering plus traditional classifiers. The lab specifically targets a core ML problem: **overfitting**, and demonstrates why a model with perfect training accuracy can still fail in the real world.

## Approach

1. **Data preparation** — Loaded Olivetti Faces (400 images, 64×64 pixels) and split into 60% train / 20% validation / 20% test *before* any modeling, with the test set held out until final evaluation.
2. **Feature extraction** — Built two classical feature descriptors:
   - **HOG** (Histogram of Oriented Gradients) — captures shape/edge structure, 1,764 dimensions per image.
   - **LBP** (Local Binary Patterns) — captures texture, 10 dimensions per image.
3. **Classifier training** — Trained **SVM (RBF kernel)** and **Random Forest** on each feature type (4 model/feature combinations total), scaling features with `StandardScaler` fit only on training data.
4. **Overfitting demonstration** — Deliberately built a 1-Nearest-Neighbor model (memorizes training data, 100% training accuracy) and compared it against a properly regularized SVM (tuned via `GridSearchCV`) to show the gap between training and validation performance.
5. **Model selection & final evaluation** — Selected the best model by **validation accuracy only**, then ran it once on the untouched test set for an honest final accuracy figure, including a per-subject accuracy breakdown.

## Results

All figures are saved in [`results/`](results/):

| Model + Features | Train Acc | Val Acc | Gap |
|---|---|---|---|
| **SVM + HOG** | 100.0% | **96.2%** | 3.7 pp |
| SVM + LBP | 88.8% | 32.5% | 56.2 pp |
| RF + HOG | 100.0% | 92.5% | 7.5 pp |
| RF + LBP | 100.0% | 41.2% | 58.8 pp |

- **Best model:** SVM + HOG, selected on validation accuracy (96.2%).
- **Final held-out test accuracy:** **95.0%** (evaluated once, after model selection was locked in).
- **Overfitting demo:** A 1-NN classifier hit 100% training accuracy but only 97.5% validation accuracy on this particular split — illustrating that a "perfect" training score is not proof of a good model. The properly tuned SVM (`C=10`, `gamma='scale'`, cross-val score 96.2%) generalized comparably despite not memorizing the training set.
- **Per-subject test accuracy:** all 40 subjects achieved ≥50% test accuracy.
- LBP features underperformed HOG substantially on this task (10-dim texture descriptor vs. 1,764-dim shape descriptor) — texture alone was insufficient to distinguish 40 different faces.

| Output | Shows |
|---|---|
| `01_sample_faces_olivetti_dataset.png` | Sample faces from 10 subjects |
| `02_pixel_value_distribution.png` | Overall pixel-intensity histogram |
| `03_hog_descriptor_visualization.png` | HOG gradient visualization on a sample face |
| `04_hog_vs_lbp_feature_distributions.png` | Feature-value distribution comparison, HOG vs LBP |
| `05_model_comparison_train_val_accuracy.png` | Train vs. validation accuracy across all 4 model/feature combos |
| `06_overfit_vs_reasonable_model_accuracy.png` | 1-NN (overfit) vs tuned SVM accuracy comparison |
| `07_confusion_matrices_overfit_vs_reasonable.png` | Side-by-side confusion matrices, overfit vs reasonable model |
| `08_learning_curve.png` | Training-size vs. accuracy learning curve |
| `09_final_test_set_confusion_matrix.png` | Final confusion matrix, SVM + HOG on the test set |
| `10_per_subject_test_accuracy.png` | Per-subject accuracy bar chart on the test set |

## Key Findings

- **Overfitting is invisible from training accuracy alone.** The 1-NN model's 100% training score came purely from memorizing each training image as its own nearest neighbor — it tells you nothing about generalization. The train/validation gap is the real signal.
- **Feature choice matters more than classifier choice here.** Both classifiers (SVM and Random Forest) performed far better with HOG than with LBP — the feature descriptor's ability to capture discriminative information capped performance more than the algorithm did.
- **Small gap ≠ small dataset problem solved.** Even with only 10 images per subject (high overfitting risk), a well-regularized SVM with HOG features generalized to 96.2% validation and 95.0% test accuracy — proper train/val/test discipline and cross-validated hyperparameter tuning controlled overfitting despite the small per-class sample size.
- **The test set must be used exactly once.** Model selection was driven entirely by validation accuracy; the test set was only touched after the winning model was locked in, giving an honest, unbiased final performance estimate.

## Technologies Used

- **Python 3**
- **scikit-learn** — `fetch_olivetti_faces`, `SVC`, `RandomForestClassifier`, `KNeighborsClassifier`, `GridSearchCV`, `StandardScaler`, `train_test_split`, `learning_curve`, confusion matrix / classification report utilities
- **scikit-image** — HOG and LBP feature extraction
- **NumPy** — array/feature manipulation
- **Matplotlib / Seaborn** — all visualizations
- **Google Colab** — development/execution environment

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells top to bottom (Runtime → Run all). The Olivetti Faces dataset downloads automatically via scikit-learn on first run.
3. Figures display inline as each cell executes; the same figures are saved separately in `results/`.

**To run locally instead:**
```bash
pip install scikit-learn scikit-image numpy matplotlib seaborn
jupyter notebook L03_A_AlexiaYChavez_ITAI_1378.ipynb
```
Requires Python 3.8+.

---
*Full reflection question answers are included in the notebook itself.*
