# ResNet50 Binary Image Classifier

A PyTorch-based binary image classification pipeline using a pretrained ResNet50 model. Supports class-specific data augmentation, training with learning rate scheduling, and evaluation with confusion matrices and ROC curves.

---

## Features

- Transfer learning with ImageNet-pretrained ResNet50
- Per-class transform support (different augmentation per class)
- Train / validation / test split (70% / 15% / 15%)
- Adam optimizer with StepLR scheduler
- Evaluation: classification report, confusion matrix, ROC curve + AUC
- Model saving and loading via state dict

---

## Requirements

Install dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

---

## Dataset Structure

The dataset should follow the standard `ImageFolder` format:

```
dataset/
├── class_0/
│   ├── img1.jpg
│   └── ...
└── class_1/
    ├── img1.jpg
    └── ...
```

Update the `DATASET_DIR` variable in the script to point to your dataset directory.

---

## Configuration

All key hyperparameters are defined at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `SEED` | `42` | Random seed for reproducibility |
| `BATCH_SIZE` | `32` | Training batch size |
| `IMG_SIZE` | `(128, 128)` | Image resize dimensions |
| `NUM_EPOCHS` | `10` | Number of training epochs |
| `LR` | `1e-3` | Initial learning rate |
| `NUM_CLASSES` | `2` | Number of output classes |
| `DATASET_DIR` | `$PATH` | Path to dataset root directory |
| `SAVE_PATH` | `resnet50_model_state_dict.pth` | Path to save model weights |

---

## Transforms

Two transform pipelines are defined and applied per class:

- **Class 0** (`transform_c1`): Resize + normalize only
- **Class 1** (`transform_c3`): Resize + random horizontal flip + random rotation + color jitter + normalize

To change which transform applies to which class, update the `transform_map` dictionary.

---

## Usage

### Train and Evaluate

```bash
python classifier.py
```

This will:
1. Load the dataset from `DATASET_DIR`
2. Split into train/val/test sets
3. Fine-tune a pretrained ResNet50
4. Save the model to `SAVE_PATH`
5. Evaluate on the test set and display plots

### Load a Saved Model

```python
from classifier import load_model
model = load_model("resnet50_model_state_dict.pth")
```

---

## Output

- **Console**: Per-epoch loss, classification report (precision, recall, F1)
- **Confusion Matrix**: Heatmap of predicted vs. true labels
- **ROC Curve**: With AUC score
<img width="640" height="480" alt="roc_curve" src="https://github.com/user-attachments/assets/7043869f-114b-4bb8-bcb4-4cd190f41a4f" />
<img width="800" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/a75dc559-f6ce-4ab8-b10f-9266556011b3" />
<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/840fa827-7968-4f3e-a535-9563a396259d" />

---

## Notes

- Training runs on GPU automatically if CUDA is available, otherwise falls back to CPU.
- The learning rate is reduced by a factor of 0.1 every 3 epochs via `StepLR`.
- The validation loader is passed to `train()` but not currently used for validation metrics during training — this can be extended to add per-epoch validation loss/accuracy tracking.
