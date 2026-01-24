import os
import re
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
import esm


MAX_LEN = 1500
SEQ_COL = "Sequence"
BATCH_SIZE = 8


class ProteinDataset(Dataset):
    
    def __init__(self, sequences):
        self.sequences = [(str(i), re.sub(r"[UZOB]", "X", str(seq)))
                          for i, seq in enumerate(sequences)]
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def pad_to_max_len(x: torch.Tensor, max_len: int = MAX_LEN) -> torch.Tensor:

    L, D = x.shape

    if L > max_len:
        x = x[:max_len]
        L = max_len

    out = torch.zeros((max_len, D), dtype=x.dtype)
    out[:L] = x
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate residue-level protein embeddings using ESM2")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()

    csv_path = args.csv_path

    # Output path: same directory, csv_name + _protein_embedding.pt
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(base_dir, f"{csv_stem}_protein_embedding.pt")

    df = pd.read_csv(csv_path)
    if SEQ_COL not in df.columns:
        raise ValueError(
            f"Column '{SEQ_COL}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = df[SEQ_COL].dropna().tolist()
    N = len(sequences)
    print(f"Total sequences: {N}")

    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    esm_model = esm_model.to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        esm_model = nn.DataParallel(esm_model)

    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    def collate_fn(batch):
        _, batch_strs, batch_tokens = batch_converter(batch)
        return batch_strs, batch_tokens.to(device)

    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    padded_embeddings = []

    with torch.no_grad():
        for batch_strs, batch_tokens in tqdm(dataloader, desc="Generating ESM embeddings"):
            padding_mask = batch_tokens != alphabet.padding_idx

            results = esm_model(tokens=batch_tokens, repr_layers=[33], return_contacts=False)
            
            residue_embeddings = results["representations"][33]
            for i in range(len(batch_strs)):
                valid_embeddings = residue_embeddings[i][1:-1][
                    padding_mask[i][1:-1]].cpu()

                padded = pad_to_max_len(valid_embeddings, MAX_LEN)
                padded_embeddings.append(padded)

    all_embeddings = torch.stack(padded_embeddings, dim=0)
    print("Final embedding shape:", all_embeddings.shape)

    os.makedirs(base_dir, exist_ok=True)
    torch.save(all_embeddings, output_file)

    print(f"Residue embeddings saved to {output_file}")


if __name__ == "__main__":
    main()