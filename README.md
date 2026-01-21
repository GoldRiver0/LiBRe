# LiBRe
> LiBRe (**Li**gand-aware **B**inding **Re**sidue Predictor) is a ligand-aware sequence-based binding residue prediction model that explicitly incorporates both residue-level information from protein sequences and ligand information.

## 🖼 Model Architecture

![LiBRe Model](docs/images/model_architecture.jpg)

## 🧩 Dependencies
- Python >= 3.8
- PyTorch >= 1.12
- numpy
- pandas
- RDKit

## 🚀 How to Use LiBRe

LiBRe requires protein residue embeddings and ligand features as inputs.
Follow the steps below to prepare the required files and run the model.

---

### Step 1. Generate Protein Residue Embeddings

Protein residue-level embeddings are generated from protein sequences using a pretrained ESM model.

```bash
python data/esm_residue_embedding.py \
  --csv_path ./data/train/train_example.csv
```

This command generates a residue-level protein embedding file (`.pt`) in the same directory as the input CSV file.

### Step 2. Generate Ligand Features from SMILES

Ligand graph features are generated from SMILES strings using RDKit and converted into PyTorch Geometric graph representations.

```bash
python data/ligand_featurizer.py \
  --csv_path ./data/train/train_example.csv
```

This command generates a ligand feature file (_ligand.pkl) in the same directory as the input CSV file.
