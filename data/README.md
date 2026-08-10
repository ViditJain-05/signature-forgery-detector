# Dataset Setup: CEDAR Signature Dataset

This project is configured to work with the **CEDAR Signature Dataset**.

## Dataset Overview
- **Source**: [Kaggle - CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
- **Subjects**: 55 individuals
- **Signatures per subject**: ~24 Genuine (original) + ~24 Forged signatures per person
- **Folder Structure**:
  - `full_org/` : Contains genuine signatures (e.g., `original_1_1.png`, ..., `original_55_24.png`)
  - `full_forg/`: Contains forged signatures (e.g., `forgeries_1_1.png`, ..., `forgeries_55_24.png`)

## Download via Kaggle CLI

```bash
# Option 1: Using Kaggle CLI
kaggle datasets download -d shreelakshmigp/cedardataset --unzip -p data/CEDAR/
```

## Expected Directory Layout

Place or extract the dataset inside `data/` so the directory tree matches:

```
signature-forgery-detector/data/
├── README.md
└── CEDAR/
    ├── full_org/
    │   ├── original_1_1.png
    │   └── ...
    └── full_forg/
        ├── forgeries_1_1.png
        └── ...
```

*Note: If no dataset is detected, `dataset.py` will automatically generate a lightweight synthetic sample dataset in `data/sample_signatures/` for quick code verification.*
