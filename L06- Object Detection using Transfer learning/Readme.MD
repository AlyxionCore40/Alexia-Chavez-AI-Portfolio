# Lab 06: Object Detection Using Transfer Learning (YOLO11)

**Course:** ITAI 1378 — Computer Vision
**Framework:** Ultralytics YOLO11
**Platform:** Google Colab (CPU)

---

## Problem Statement

Training an object detector from scratch requires huge amounts of labeled data and compute. Transfer learning solves this by starting from a model already trained on a large general dataset (YOLO11, pretrained on COCO's 80 object classes) and adapting it to a new task instead. This lab investigates *how* to adapt it — comparing three strategies (zero-shot use, frozen-backbone feature extraction, and full fine-tuning) to understand the practical tradeoffs between training cost and detection accuracy.

## Approach

Four detection runs were evaluated on the `coco8` dataset (a small 8-image COCO sample), using mAP, precision, and recall as metrics:

1. **Zero-shot baseline** — Ran the pretrained YOLO11n model with no additional training at all, as a "before transfer learning" reference point.
2. **Feature extraction (frozen backbone)** — Froze the first 10 layers (the backbone) and trained only the detection head for 10 epochs, so the model reuses its pretrained visual features unchanged.
3. **Full fine-tuning** — Unfroze the entire model (`freeze=0`) and trained all layers for 10 epochs, letting the backbone itself adapt to the new data.
4. **Custom experiment** — Repeated fine-tuning with a different configuration (`freeze=5`, lower learning rate `lr0=0.001`) to test a middle ground between the two extremes above.

All four runs were evaluated with the same validation split and compared side-by-side in a single table and bar chart.

## Results

All figures are saved in [`results/`](results/):

| Output | Shows |
|---|---|
| `01_zero_shot_baseline_prediction.png` | Zero-shot YOLO11 detection on a sample image, no training |
| `02_mAP_comparison_across_approaches.png` | Bar chart comparing mAP50-95 and mAP50 across all four runs |
| `03_transfer_learning_strategies_diagram.png` | Diagram of the three core strategies compared (which layers are frozen vs. trainable) — generated to document the notebook's core comparison, since it wasn't already visualized as a diagram |

**Metrics across all four runs:**

| Approach | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|
| Zero-shot baseline | 0.630 | 0.846 | 0.570 | 0.850 |
| Feature extraction (frozen backbone) | 0.613 | 0.820 | 0.634 | 0.850 |
| **Full fine-tuning** | **0.647** | **0.899** | **0.699** | **0.879** |
| Custom experiment (freeze=5, lr0=0.001) | 0.633 | 0.842 | 0.659 | 0.850 |

- **Best overall performer:** full fine-tuning, with the highest scores on every metric (mAP50-95 0.647, mAP50 0.899, precision 0.699, recall 0.879).
- **Feature extraction underperformed the zero-shot baseline** on mAP50-95 (0.613 vs. 0.630) — with only 10 epochs on this tiny 8-image dataset, freezing the backbone and training just the head didn't outperform not training at all.
- **The custom experiment (freeze=5, lower learning rate) landed between the two extremes**, close to but slightly above the zero-shot baseline, and below full fine-tuning.

## Key Findings

- On a very small, in-distribution dataset like `coco8` (which is drawn from the same data YOLO11 was already pretrained on), full fine-tuning had the most capacity to adapt and won on every metric, but the margin over the zero-shot baseline was modest given only 10 training epochs.
- Freezing more of the network (feature extraction) does not automatically help — here it actually scored slightly worse than doing no training at all, showing that "frozen backbone + short training" isn't a free win; it depends heavily on how well the pretrained features already match the target data.
- Precision and recall trade off differently by strategy: full fine-tuning improved both together, while other strategies improved recall/precision unevenly — reinforcing that mAP alone doesn't tell the whole story for a specific deployment (e.g., an access-control system would care more about precision than raw mAP).
- Learning rate and freeze depth interact: the custom experiment's lower learning rate (0.001 vs. the default 0.01) combined with partial freezing (5 layers) produced a middle-ground result, illustrating that fine-tuning hyperparameters need to be tuned together, not independently.
- Because `coco8` overlaps with YOLO11's original training distribution, these results likely overstate how well any of the transfer strategies would generalize to a genuinely novel domain (e.g., a custom species or product) — a real project should validate on out-of-distribution data.

## Technologies Used

- **Python 3**
- **Ultralytics YOLO11** (`ultralytics` package) — pretrained detection model, training (`model.train`), validation (`model.val`), and inference (`model.predict`)
- **OpenCV** (`cv2`) — image color-space conversion for visualization
- **Pandas** — building the cross-run comparison table
- **Matplotlib** — bar chart visualization of results
- **Google Colab** — execution environment (CPU)

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run all cells top to bottom (Runtime → Run all). The `ultralytics` package, YOLO11 pretrained weights, and the `coco8` dataset all download automatically on first use — no manual setup required.
3. To use your own dataset instead of `coco8`, replace `DATA_YAML` with the path to your own `data.yaml` in YOLO format.
4. Figures display inline as each cell runs; the same figures are saved separately in `results/`.

**To run locally instead:**
```bash
pip install ultralytics opencv-python-headless pandas matplotlib
jupyter notebook L06_Alexia_Y_Chavez_ITAI_1378.ipynb
```
Requires Python 3.8+. A GPU is recommended for the training cells but not required — this run completed on CPU.

---
*Full reflection question answers (10 questions, including the midterm project pitch) are included in the notebook itself.*
