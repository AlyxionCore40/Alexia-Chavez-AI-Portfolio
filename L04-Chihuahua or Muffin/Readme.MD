# Lab 04: Chihuahua or Muffin? — Fully-Connected Neural Network

**Course:** ITAI 1378
**Framework:** PyTorch
**Platform:** Google Colab

---

## Problem Statement

"Chihuahua or muffin" is a well-known, deliberately tricky image classification problem: photos of chihuahua faces and blueberry muffins can look surprisingly similar in shape, color, and texture, making it a good test of whether a model is learning real distinguishing features or just superficial ones. This lab builds a basic **fully-connected (dense) neural network** from scratch in PyTorch to classify images into these two categories, as an introduction to the deep learning workflow (build → load data → train → evaluate) before moving on to convolutional approaches.

## Approach

1. **Data preparation** — Loaded a labeled image dataset via `ImageFolder`: 120 training images and 30 validation images, split across `chihuahua` and `muffin` subfolders.
2. **Preprocessing** — Resized all images to 224×224 and normalized pixel values (mean/std = 0.5) via `torchvision.transforms`.
3. **Model architecture** — Built `MySkynet`, a custom fully-connected network that flattens each 224×224×3 image into a 150,528-length vector and passes it through three hidden layers (128 → 64 → 32 units, ReLU activations) to a final 2-unit output layer with softmax.
4. **Training** — Trained for 5 epochs using Cross-Entropy loss and plain SGD (learning rate 0.01), tracking loss and accuracy on both training and validation splits each epoch.
5. **Evaluation** — Ran the trained model on the full validation set and visualized individual predictions to inspect which images the model got right or wrong.

## Results

All figures are saved in [`results/`](results/):

| Output | Shows |
|---|---|
| `01_sample_images_chihuahua_muffin.png` | Example training images from both classes |
| `02_training_validation_loss_accuracy_curves.png` | Training/validation loss and accuracy over 5 epochs |
| `03_sample_predictions_validation_set.png` | Grid of validation images with predicted vs. true labels |
| `04_model_architecture_diagram.png` | Visual diagram of the network's layer-by-layer architecture (generated from the model's actual layer definitions) |

**Training log (actual run, 5 epochs):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.677 | 73.3% | 0.649 | 76.7% |
| 2 | 0.620 | 85.0% | 0.599 | 90.0% |
| 3 | 0.551 | 89.2% | 0.540 | 90.0% |
| 4 | 0.488 | 93.3% | 0.505 | 96.7% |
| 5 | 0.437 | 95.0% | 0.485 | 83.3% |

- **Peak validation accuracy:** 96.7% (epoch 4).
- **Final validation accuracy (full evaluation pass):** **83.3%** (25/30 correct).
- Training loss and accuracy improved smoothly and consistently every epoch, but validation accuracy was noisy — it dropped from 96.7% to 83.3% in the final epoch, on a validation set of only 30 images (so each misclassified image swings the accuracy by ~3.3 percentage points).

## Key Findings

- A plain fully-connected network can learn this task reasonably well (83.3% final validation accuracy after just 5 epochs) despite discarding all spatial structure in the image by flattening it into a single vector.
- Flattening a 224×224×3 image into 150,528 raw pixel values means the very first layer alone has over 19 million trainable parameters — this is an inefficient way to represent an image, since it ignores the fact that nearby pixels are related.
- Training accuracy climbed monotonically while validation accuracy fluctuated — a sign that with only 120 training images and no data augmentation, the model is sensitive to which images land in each batch and starting to overfit slightly toward the end of training.
- Because the final epoch's validation accuracy (83.3%) is lower than an earlier epoch's (96.7% at epoch 4), the very last trained weights are not necessarily the "best" weights — this motivates checkpointing/selecting the best-validation-epoch model rather than always using the final one.
- This lab's dense-network baseline is a useful comparison point for a later CNN-based version of the same task, to isolate how much benefit convolutional layers provide over a flat architecture.

## Technologies Used

- **Python 3**
- **PyTorch** (`torch`, `torchvision`) — model definition (`nn.Module`, `nn.Linear`), training loop, data loading/preprocessing (`ImageFolder`, `DataLoader`, `transforms`)
- **PIL (Pillow)** — image loading for the visualization cells
- **Matplotlib** — training curves and prediction visualizations
- **Google Colab** — development/execution environment

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells top to bottom. The first cell clones the dataset repository (`workshop-chihuahua-vs-muffin`) directly from GitHub, so no manual upload is needed.
3. A GPU runtime will speed up training but isn't required — the model will fall back to CPU automatically.
4. Figures display inline during/after training; the same figures are saved separately in `results/`.

**To run locally instead:**
```bash
pip install torch torchvision pillow matplotlib
git clone https://github.com/patitimoner/workshop-chihuahua-vs-muffin.git
jupyter notebook L04_Chihuahua_or_Muffin_ITAI_1378.ipynb
```
Requires Python 3.8+.
