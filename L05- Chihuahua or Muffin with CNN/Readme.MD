# Lab 05: Chihuahua vs Muffin Classifier — Convolutional Neural Network

**Course:** ITAI 1378
**Framework:** PyTorch
**Platform:** Google Colab (CPU)

---

## Problem Statement

The "chihuahua or muffin" task is a well-known, deliberately hard image classification problem — the two classes can look startlingly similar in shape and color, making it a good stress test for a model's ability to learn real visual features rather than superficial cues. This lab builds on an earlier workshop that used a plain (fully-connected) neural network for this task, and replaces it with a **Convolutional Neural Network (CNN)** to test whether learning spatial/local features via convolution improves classification over a flat NN.

## Approach

1. **Data preparation** — Loaded a labeled image dataset (65 chihuahua / 55 muffin training images, 17 chihuahua / 13 muffin validation images) via `ImageFolder`, resized to 64×64 for efficient CPU training.
2. **Augmentation** — Applied random horizontal flips and ±10° rotations to training data only; validation data was resized/normalized without augmentation, to get an honest measure of generalization.
3. **Model architecture** — Built a custom CNN (`ChihuahuaMuffinCNN`) with three convolutional blocks (Conv2d → ReLU → MaxPool2d, channels 3→32→64→128) followed by a fully-connected classifier (8192 → 512 → 2) with 50% dropout for regularization.
4. **Training** — Trained for 10 epochs using Cross-Entropy loss and the Adam optimizer (lr = 0.001), tracking loss and accuracy on both the training and validation splits every epoch.
5. **Evaluation** — Ran the trained model over the full validation set and visualized individual predictions (correct in green, incorrect in red) to inspect failure cases directly.

## Results

All figures are saved in [`results/`](results/):

| Output | Shows |
|---|---|
| `01_sample_predictions_chihuahua_muffin.png` | Grid of validation images with predicted vs. true label (green = correct, red = wrong) |
| `02_training_validation_loss_accuracy_curves.png` | Training/validation loss and accuracy over 10 epochs |
| `03_model_architecture_diagram.png` | Visual diagram of the CNN's layer-by-layer architecture (generated from the model's actual layer definitions) |

**Training log (actual run, 10 epochs):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.835 | 43.3% | 0.672 | 46.7% |
| 2 | 0.662 | 55.8% | 0.646 | 76.7% |
| 3 | 0.594 | 81.7% | 0.488 | 83.3% |
| 4 | 0.403 | 85.8% | 0.257 | 90.0% |
| 5 | 0.343 | 85.8% | 0.245 | 93.3% |
| 6 | 0.263 | 90.0% | 0.183 | 96.7% |
| 7 | 0.184 | 91.7% | 0.226 | 96.7% |
| 8 | 0.190 | 94.2% | 0.172 | 96.7% |
| 9 | 0.102 | 96.7% | 0.179 | 96.7% |
| 10 | 0.145 | 93.3% | 0.162 | 90.0% |

- **Peak validation accuracy:** 96.7% (epochs 6, 7, 8, and 9).
- **Final validation-set evaluation** (via `evaluate_model` after training): **90.0%** accuracy.
- Both training and validation loss trend downward overall, though validation loss is noisy given the very small validation set (30 images) — a few misclassified images swing the accuracy several percentage points.

> Note: the notebook's separate training-curve plotting cell (`02_training_validation_loss_accuracy_curves.png`) uses a slightly different set of numbers than the executed training log above (likely example/placeholder values left over from drafting). The table above reflects the actual printed output of the training run in this notebook.

## Key Findings

- The CNN reaches substantially higher validation accuracy (peaking at 96.7%) than the flat NN baseline from the prior lab (which reportedly peaked around 83%), supporting the idea that convolutional layers' ability to learn local spatial patterns (edges, textures, shapes) is well suited to this image task.
- The gap between peak (96.7%) and final (90.0%) validation accuracy shows some instability/overfitting risk late in training on such a small dataset (120 training images) — accuracy isn't monotonically improving, and the last epoch's validation accuracy actually dropped from the epoch before.
- Data augmentation (flips, small rotations) on the training set only was used specifically to reduce overfitting risk given the very small dataset size.
- Misclassified examples in the prediction grid are the most informative artifact for understanding *why* the model fails — visually inspecting which chihuahuas were called muffins (or vice versa) is more useful than accuracy alone for judging real-world readiness.

## Technologies Used

- **Python 3**
- **PyTorch** (`torch`, `torchvision`) — model definition, training loop, data loading/augmentation (`ImageFolder`, `DataLoader`, `transforms`)
- **torchsummary** — model summary inspection
- **tqdm** — training progress bars
- **NumPy** — array operations
- **Matplotlib** — training curves and prediction visualizations
- **Google Colab** — execution environment (CPU)

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells top to bottom. The first cell clones the dataset repository (`workshop-chihuahua-vs-muffin`) directly from GitHub, so no manual upload is needed.
3. Training runs on CPU by default (64×64 images chosen specifically for CPU efficiency); switch to a GPU runtime and increase `input_height`/`input_width` to 224 for higher accuracy if desired.
4. Figures display inline during/after training; the same figures are saved separately in `results/`.

**To run locally instead:**
```bash
pip install torch torchvision torchsummary tqdm numpy matplotlib
git clone https://github.com/patitimoner/workshop-chihuahua-vs-muffin.git
jupyter notebook Lab05_CNN_Chihuahua_Muffin.ipynb
```
Requires Python 3.8+.

---
*Full reflection question answers are included in the notebook itself.*
