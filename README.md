
# Colorectal Polyp Classification Using ResNet50

A research prototype exploring deep learning for binary classification of colorectal polyp images into adenoma and hyperplastic classes using PyTorch and ResNet50.

## Project Overview

This project investigates transfer learning and fine-tuning strategies for colorectal polyp image classification. It compares different training configurations, including full fine-tuning, a frozen feature extractor, offline data augmentation, and class-weighted loss.

The project is intended for academic research and experimentation. It is not a clinically validated diagnostic system.

## Research Objectives

- Explore ResNet50 for binary colorectal polyp image classification.
- Compare different transfer-learning configurations.
- Investigate offline image augmentation with a small dataset.
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Document limitations associated with small datasets and overfitting.

## Classes

- `adenoma`
- `hyperplastic`

Class labels should be assigned using appropriately verified reference diagnoses. This repository does not contain the original clinical dataset.

## Experiments

The following are preliminary validation results from experiments conducted during development.

| Experiment | Configuration | Validation accuracy |
|---|---|---:|
| Experiment 2 | ResNet50 fine-tuning with a lower learning rate | 52.17% |
| Experiment 3 | Frozen ResNet50 feature extractor | 47.83% |
| Experiment 4 | Fine-tuned ResNet50 with offline augmentation | 65.22% |
| Experiment 5 | Frozen ResNet50 with offline augmentation | 56.52% |
| Experiment 6 | Frozen ResNet50 with class-weighted loss | 34.78% |

Experiment 4 achieved the highest selected-checkpoint validation accuracy among these experiments. Its training accuracy reached 100%, indicating a substantial risk of overfitting.

These results are preliminary and should not be interpreted as evidence of clinical effectiveness. Experiments used a small validation set of 23 images, and training configurations were not identical across all experiments. The validation set was also used during model selection.

## Methodology

1. Organize images into adenoma and hyperplastic classes.
2. Split the available dataset into training and validation sets, grouping by patient where possible to reduce patient-level leakage.
3. Apply augmentation to training images only.
4. Train ResNet50 using different transfer-learning configurations.
5. Evaluate selected checkpoints using validation metrics.
6. Compare performance and document limitations.

Augmented images are derived from existing images and do not increase the number of independent patients.

## Project Structure

```text
Polyp_Classification/
├── train_experiment2.py
├── train_experiment3.py
├── train_experiment4.py
├── train_experiment5.py
├── train_experiment6.py
├── create_augment_dataset.py
├── create_manifest.py
├── split_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above lists selected project files. Local datasets, patient-related metadata, checkpoints, and private experiment outputs are intentionally excluded from the public repository.

## Installation

Use Python and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the general dependencies:

```bash
python -m pip install -r requirements.txt
```

Install PyTorch and torchvision separately using the official installation instructions, selecting compatible versions for your operating system and CUDA environment:

https://pytorch.org/get-started/locally/

Verify the installation:

```python
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Dataset and Privacy

The original dataset is not included because it contains clinical images collected for research. Access to the data must follow applicable hospital permissions, ethics approvals, and data-sharing agreements.

Users must provide their own appropriately authorized dataset and adapt file paths and dataset preparation settings as needed.

## Limitations

- Small number of independent patient cases.
- Limited validation sample size.
- Potential overfitting and unstable validation metrics.
- Results may not generalize across hospitals, endoscopy systems, or imaging modalities.
- A separate, patient-level held-out test set is needed for a stronger final evaluation.

## Intended Use

For research and educational purposes only. This project is not intended to replace clinical assessment, histopathology, or the judgment of qualified healthcare professionals.

## License

A license will be added after the repository owner selects the appropriate terms.
```
