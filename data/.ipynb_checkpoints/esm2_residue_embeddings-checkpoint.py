import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import esm
import re
import os
import pandas as pd
from tqdm import tqdm

batch_size = 32  

data = pd.read_csv('./data/Test/COACH420.csv')
protein_sequence = data['padded_Sequence'].to_list()


esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
emb_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(emb_device)
esm_model = nn.DataParallel(esm_model)

esm_model.eval()  
batch_converter = alphabet.get_batch_converter()

class ProteinDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [(str(i), re.sub(r"[UZOB]", "X", seq)) for i, seq in enumerate(sequences)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch):
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(emb_device)
    return batch_strs, batch_tokens


protein_dataset = ProteinDataset(protein_sequence)
protein_dataloader = DataLoader(protein_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)


all_embeddings = []


with torch.no_grad():
    for batch_strs, batch_tokens in tqdm(protein_dataloader, desc="Processing batches"):
        padding_mask = batch_tokens != alphabet.padding_idx
        

        results = esm_model(batch_tokens, repr_layers=[30], return_contacts=False)
        residue_embeddings = results["representations"][30]
        
        for i in range(len(batch_strs)):
            valid_embeddings = residue_embeddings[i][1:-1][padding_mask[i][1:-1]].cpu()
            all_embeddings.append(valid_embeddings)

output_file = './data/Test/HOLO4K_residue_embeddings_esm2.pt'
torch.save(all_embeddings, output_file)
print(f"Residue embeddings saved to {output_file}")