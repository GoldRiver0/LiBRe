import os
import re
import sys
from tqdm import tqdm
import random
import pickle
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from functools import partial

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)


def fbeta_from_pr(prec, rec, beta=2.0, eps=1e-12):
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec + eps)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser("LiBRe Args")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--train_emb", type=str, required=True)
    parser.add_argument("--train_ligand", type=str, default=None)

    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--test_emb", type=str, default=None)
    parser.add_argument("--test_ligand", type=str, default=None)

    parser.add_argument("--use_cnn_lstm", action="store_true")
    parser.add_argument("--use_ligand", action="store_true")
    parser.add_argument("--residue_input_dim", type=int, default=1280)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_data(csv_path, embedding_path, ligand_pkl_path=None):
    # CSV data load
    data = pd.read_csv(csv_path)
    pdb_id = data['PDB_ID'].to_list()
    ligand_code = data['Ligand_code'].to_list()
    sequence = data['padded_Sequence'].to_list()
    chain = data['Chain'].to_list()
    origin_len = data['Sequence_length'].to_list()
    label = data['label_sequence'].to_list()
    
    # Embeddings load
    embeddings = torch.load(embedding_path)
    print(f"Loaded embeddings from {embedding_path}")
    
    # ligand data load
    if ligand_pkl_path and os.path.exists(ligand_pkl_path):
        with open(ligand_pkl_path, 'rb') as f:
            ligand_data = pickle.load(f)
        print(f"Loaded ligand data from {ligand_pkl_path}")
    else:
        ligand_data = data['Ligand_smiles'].to_list() if 'Ligand_smiles' in data else None
        print(f"Ligand data file not found, using Ligand_smiles from {csv_path}")

    return pdb_id, sequence, ligand_code, chain, origin_len, label, embeddings, ligand_data


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    symbol = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
    degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "other",
    ]

    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol)
    results += one_of_k_encoding(atom.GetDegree(), degree)
    results += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    results += one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType)
    results += [atom.GetIsAromatic()]

    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
            results += [atom.HasProp("_ChiralityPossible")]
        except Exception:
            results += [False, False, atom.HasProp("_ChiralityPossible")]

    return results


def adjacent_matrix(mol):
    return np.array(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)


def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError("SMILES cannot be parsed")

    atom_feat = np.zeros((mol.GetNumAtoms(), 40), dtype=np.float32)
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)

    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def smiles_to_data(smiles):
    atom_feat, adj_matrix = mol_features(smiles)
    edges = np.nonzero(adj_matrix)
    edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
    x = torch.tensor(atom_feat, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def get_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp + 1e-8)
    return 0.0


class LiBReDataset(Dataset):
    def __init__(self, sequences, origin_len, labels, embedding_tensor, ligands):
        self.origin_len = origin_len
        self.ligands = ligands
        self.labels = [list(lbl) for lbl in labels]
        self.res_embedding = embedding_tensor
        self.sequences = [" ".join(list(re.sub(r"[UZOB]", "X", str(seq)))) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            int(self.origin_len[idx]),
            self.ligands[idx],
            self.labels[idx],
            self.res_embedding[idx],
        )
        

def collate_fn(batch):
    sequences, origin_len, ligands, labels, res_embeddings = zip(*batch)

    embeddings = torch.stack(res_embeddings, dim=0)
    max_len = embeddings.size(1)

    padded_labels = [lbl + ["0"] * (max_len - len(lbl)) for lbl in labels]
    padded_labels = torch.tensor([[int(x) for x in lbl] for lbl in padded_labels], dtype=torch.long)

    if ligands[0] is None:
        ligands_batch = None
    elif isinstance(ligands[0], Data):
        ligands_batch = Batch.from_data_list(list(ligands))
    else:
        ligands_batch = Batch.from_data_list([smiles_to_data(sm) for sm in ligands])

    return embeddings, list(origin_len), ligands_batch, padded_labels


def create_dataloader(sequence, 
                      origin_len, 
                      label, 
                      embeddings, 
                      ligand, 
                      batch_size, 
                      shuffle=True):
    dataset = LiBReDataset(sequence, origin_len, label, embeddings, ligand)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(collate_fn))

def _subset_by_index(obj, indices):
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        return [obj[i] for i in indices]

    if torch.is_tensor(obj):
        return obj[indices]

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj[indices]
    except Exception:
        pass

    return [obj[i] for i in indices]

def make_train_val_loaders(
    sequence,
    origin_len,
    label,
    embeddings,
    ligand,
    train_bs,
    eval_bs,
    val_ratio=0.3,
    seed=42,
    shuffle=True):
    n = len(sequence)
    g = torch.Generator().manual_seed(seed)

    indices = torch.randperm(n, generator=g).tolist() if shuffle else list(range(n))
    val_size = int(n * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_sequence = _subset_by_index(sequence, train_idx)
    train_origin_len = _subset_by_index(origin_len, train_idx)
    train_label = _subset_by_index(label, train_idx)
    train_embeddings = _subset_by_index(embeddings, train_idx)
    train_ligand = _subset_by_index(ligand, train_idx)

    val_sequence = _subset_by_index(sequence, val_idx)
    val_origin_len = _subset_by_index(origin_len, val_idx)
    val_label = _subset_by_index(label, val_idx)
    val_embeddings = _subset_by_index(embeddings, val_idx)
    val_ligand = _subset_by_index(ligand, val_idx)

    train_loader = create_dataloader(
        train_sequence,
        train_origin_len,
        train_label,
        train_embeddings,
        train_ligand,
        batch_size=train_bs,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_sequence,
        val_origin_len,
        val_label,
        val_embeddings,
        val_ligand,
        batch_size=eval_bs,
        shuffle=False,
    )
    return train_loader, val_loader

def print_split_line(width=84):
    print("-" * width)

def print_eval_header():
    print("{:<7}{:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<10} {:<10}".format(
        "", "ACC", "PREC", "REC", "F1", "F2", "MCC", "ROC_AUC", "PR_AUC"
    ))

def print_eval_row(split, m):
    print("{:<7}{:<9.4f} {:<9.4f} {:<9.4f} {:<9.4f} {:<9.4f} {:<9.4f} {:<10.4f} {:<10.4f}".format(
        split,
        m["ACC"], m["PREC"], m["REC"],
        m["F1"], m["F2"], m["MCC"],
        m["ROC_AUC"], m["PR_AUC"]
    ))

# -------------------------------------------------------------------------------------------------------- #

def nt_xent_loss(embeddings, labels, temperature=1.0):
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    embeddings = embeddings.view(-1, embeddings.size(-1))
    labels = labels.view(-1)

    sim = torch.matmul(embeddings, embeddings.T) / temperature
    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(mask, -float("inf"))

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~mask)
    exp_sim = torch.exp(sim)

    sum_all = exp_sim.sum(dim=1)
    sum_pos = (exp_sim * pos_mask.float()).sum(dim=1)

    return (-torch.log((sum_pos + 1e-8) / (sum_all + 1e-8))).mean()

def train(dataloader, model, criterion, optimizer, epoch, epochs, use_contrastive=False, log_every=10):
    model.train()
    total_loss = 0.0
    n_steps = len(dataloader)

    for step, batch in enumerate(dataloader, start=1):
        res_embeddings, origin_len, ligand_graphs, padded_labels = batch

        res_embeddings = res_embeddings.to(DEVICE)
        padded_labels = padded_labels.float().to(DEVICE)
        if ligand_graphs is not None:
            ligand_graphs = ligand_graphs.to(DEVICE)

        binding_embedding, affinity_score = model(res_embeddings, ligand_graphs)
        affinity_score = affinity_score.squeeze(-1)

        pred_score, pred_emb, true_label = [], [], []
        for j, length in enumerate(origin_len):
            pred_score.append(affinity_score[j, :length])
            pred_emb.append(binding_embedding[j, :length])
            true_label.append(padded_labels[j, :length])

        pred_score = torch.cat(pred_score)
        pred_emb = torch.cat(pred_emb)
        true_label = torch.cat(true_label)

        bce_loss = criterion(pred_score, true_label).mean()
        contrastive_loss = (
            nt_xent_loss(pred_emb, true_label, temperature=0.2)
            if use_contrastive
            else torch.zeros((), device=pred_score.device)
        )
        loss = bce_loss if not use_contrastive else 0.5 * bce_loss + 0.5 * contrastive_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / step

        if (step % log_every == 0) or (step == 1) or (step == n_steps):
            print(
                f"[Train] Epoch {epoch}/{epochs} | Batch {step}/{n_steps} | "
                f"avg_loss={avg_loss:.4f}",
                end="\r",
                flush=True,
            )
    print()
    return total_loss / max(n_steps, 1)

@torch.no_grad()
def evaluate(dataloader, model, threshold=0.6):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in dataloader:
        res_embeddings, origin_len, ligand_graphs, padded_labels = batch

        res_embeddings = res_embeddings.to(DEVICE)
        padded_labels = padded_labels.float().to(DEVICE)

        if ligand_graphs is not None:
            ligand_graphs = ligand_graphs.to(DEVICE)

        _, affinity_score = model(res_embeddings, ligand_graphs)
        affinity_score = affinity_score.squeeze(-1)

        pred_score, true_label = [], []
        for j, length in enumerate(origin_len):
            pred_score.append(affinity_score[j, :length])
            true_label.append(padded_labels[j, :length])

        pred_score = torch.cat(pred_score)
        true_label = torch.cat(true_label)

        pred_prob = torch.sigmoid(pred_score)

        all_probs.append(pred_prob.cpu().numpy())
        all_labels.append(true_label.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_from_pr(prec, rec, beta=2.0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "ACC": acc,
        "PREC": prec,
        "REC": rec,
        "F1": f1,
        "F2": f2,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }