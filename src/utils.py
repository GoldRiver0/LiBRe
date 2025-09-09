import re
import os
import sys
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    """
    Argument parser to get model and training configurations
    """
    parser = argparse.ArgumentParser(description="Protein-Ligand Binding Residue Predictor Training")

    parser.add_argument("--use_cnn_lstm", type=bool, default=False, help="Use CNN+LSTM for residue encoding (True/False)")
    parser.add_argument("--use_ligand", type=bool, default=False, help="Use ligand information in the model (True/False)")
    parser.add_argument("--use_contrastive", type=bool, default=False, help="Enable contrastive learning in training (True/False)")

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


def create_dataloader(sequence, origin_len, label, embeddings, ligand, batch_size):
    dataset = BRPDataset(sequence, origin_len, label, embeddings, ligand)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn))


# Functions for SMILES to graph conversion
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]
    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency, dtype=np.float32)


def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError("SMILES cannot be parsed!")
    atom_feat = np.zeros((mol.GetNumAtoms(), 40))  # 40 features per atom
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def smiles_to_data(smiles):
    atom_feat, adj_matrix = mol_features(smiles)
    edge_index = np.array(np.nonzero(adj_matrix), dtype=np.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(atom_feat, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data


def get_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-8)  # small epsilon to avoid zero division
    else:
        specificity = 0.0  # fallback if one class is missing (e.g., all 1s or 0s)
    
    return specificity


class BRPDataset(Dataset):
    def __init__(self, sequences, origin_len, labels, embedding_file, ligands):
        # 시퀀스를 전처리하여 저장
        self.sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        self.origin_len = origin_len
        self.ligands = ligands
        self.labels = [list(label) for label in labels]  # 문자열 레이블을 리스트로 변환
        self.res_embedding = embedding_file
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        origin_len = self.origin_len[idx]
        ligand = self.ligands[idx]
        label = self.labels[idx]
        
        res_embedding = self.res_embedding[idx]  # 저장된 임베딩을 불러옴
        
        return sequence, origin_len, ligand, label, res_embedding


def collate_fn(batch):
    sequences, origin_len, ligands, labels, res_embeddings = zip(*batch)
    
    max_len = max(emb.size(0) for emb in res_embeddings)

    embeddings = torch.stack(res_embeddings)
        
    # 레이블 패딩
    padded_labels = [
        label + ['0'] * (max_len - len(label)) for label in labels
    ]
    padded_labels = torch.tensor([[int(x) for x in lbl] for lbl in padded_labels])

    # Process ligand graphs
    if isinstance(ligands[0], Data):
        # Already preprocessed graphs
        ligands = Batch.from_data_list(ligands)
    else:
        print("Converting SMILES to graph...")
        ligands = Batch.from_data_list([smiles_to_data(ligand) for ligand in ligands])
    
    return embeddings, origin_len, ligands, padded_labels


def find_best_threshold(all_preds, all_labels, device, thresholds=np.arange(0.1, 1.0, 0.001)):
    
    all_preds = torch.tensor(all_preds, device=device)
    all_labels = torch.tensor(all_labels, device=device).long()  # 0 또는 1

    # 임계값들 벡터화
    thresholds_tensor = torch.tensor(thresholds, device=device).view(1, -1)  # [1, T]
    pred_matrix = (all_preds.view(-1, 1) > thresholds_tensor).long()         # [N, T]

    # TP: 예측=1, 실제=1
    true_positives = ((pred_matrix == 1) & (all_labels.view(-1, 1) == 1)).sum(dim=0)
    # FN: 예측=0, 실제=1
    false_negatives = ((pred_matrix == 0) & (all_labels.view(-1, 1) == 1)).sum(dim=0)

    recalls = true_positives.float() / (true_positives + false_negatives + 1e-8)  # [T]
    best_idx = torch.argmax(recalls)
    best_threshold = thresholds[best_idx.item()]

    return best_threshold

def cpu_find_best_threshold(all_preds, all_labels, device, thresholds=np.arange(0.1, 1.0, 0.001)):
    preds_tensor = torch.tensor(all_preds, device=device)
    labels_tensor = torch.tensor(all_labels, device=device).long()

    best_threshold = 0.001
    best_f1 = 0.0

    for threshold in thresholds:
        binarized = (preds_tensor > threshold).long()

        tp = ((binarized == 1) & (labels_tensor == 1)).sum().item()
        fp = ((binarized == 1) & (labels_tensor == 0)).sum().item()
        fn = ((binarized == 0) & (labels_tensor == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def cpu_find_best_threshold_all_metrics(all_preds, all_labels, device, thresholds=np.arange(0.1, 1.0, 0.001)):
    preds_tensor = torch.tensor(all_preds, device=device)
    labels_tensor = torch.tensor(all_labels, device=device).long()

    # 결과 저장용 dict
    best_metrics = {
        "precision": {"value": 0.0, "threshold": 0.0},
        "recall": {"value": 0.0, "threshold": 0.0},
        "f1": {"value": 0.0, "threshold": 0.0},
        "specificity": {"value": 0.0, "threshold": 0.0},
        "mcc": {"value": -1.0, "threshold": 0.0},  # MCC는 -1~1 사이
        "accuracy": {"value": 0.0, "threshold": 0.0},
    }

    for threshold in thresholds:
        binarized = (preds_tensor > threshold).long()

        tp = ((binarized == 1) & (labels_tensor == 1)).sum().item()
        tn = ((binarized == 0) & (labels_tensor == 0)).sum().item()
        fp = ((binarized == 1) & (labels_tensor == 0)).sum().item()
        fn = ((binarized == 0) & (labels_tensor == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        f1 = 5 * precision * recall / (4 * precision + recall + 1e-8)
        
        # MCC 계산은 numpy 배열로 해야 정확함
        mcc = matthews_corrcoef(labels_tensor.cpu().numpy(), binarized.cpu().numpy())

        # 각 metric에 대해 최적 threshold 갱신
        for name, value in zip(
            ["precision", "recall", "f1", "specificity", "mcc", "accuracy"],
            [precision, recall, f1, specificity, mcc, accuracy]
        ):
            if value > best_metrics[name]["value"]:
                best_metrics[name]["value"] = value
                best_metrics[name]["threshold"] = threshold

    return best_metrics



def nt_xent_loss(embeddings, labels, temperature=1.0):
    """
    Computes the NT-Xent Loss (SimCLR contrastive loss) with memory optimizations:
    - Removes self-similarity by masking diagonal entries.
    - Uses a single exponentiation on the similarity matrix.
    
    Args:
        embeddings: Tensor of shape [batch_size, embedding_dim]
        labels: Tensor of shape [batch_size]
        temperature: Scaling factor for contrastive loss
    
    Returns:
        loss: Scalar tensor representing the contrastive loss
    """
    device = embeddings.device

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    # Flatten embeddings and labels for generality
    embeddings = embeddings.view(-1, embeddings.size(-1))  # [N, embedding_dim]
    labels = labels.view(-1)  # [N]
    
    batch_size = embeddings.size(0)
    
    # Compute similarity matrix scaled by temperature
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # [N, N]

    # Create a mask for self-similarity (diagonal elements)
    diag_mask = torch.eye(batch_size, device=device).bool()
    # Exclude self-similarity by setting diagonal to -inf.
    similarity_matrix = similarity_matrix.masked_fill(diag_mask, -float('inf'))
    
    # Create positive mask based on label equality & remove self-comparison
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & (~diag_mask)
    
    # Compute exponentiated similarities once
    exp_sim_matrix = torch.exp(similarity_matrix)  # [N, N]
    
    # Denominator: sum over all (masked) similarities
    sum_all = exp_sim_matrix.sum(dim=1, keepdim=True)  # [N, 1]
    
    # Numerator: sum over positive similarities only
    sum_pos = (exp_sim_matrix * positive_mask.float()).sum(dim=1, keepdim=True)  # [N, 1]
    
    # Avoid division by zero
    eps = 1e-8
    loss = -torch.log((sum_pos + eps) / (sum_all + eps))
    
    # Mean loss over all samples
    return loss.mean()


def train(num_epochs, dataloader, model, criterion, optimizer, model_device, lambda_contrastive=0.5, use_contrastive=True):
    
    model.train()
    epoch_losses = []
    
    train_loss = 0.0  # 에폭별로 loss를 추적
    num_batches = len(dataloader)  # 배치 개수 확인

    for batch_idx, batch in enumerate(dataloader):
        res_embeddings, origin_len, ligand_graphs, padded_labels = batch
        
        res_embeddings = res_embeddings.to(model_device)
        ligand_graphs = ligand_graphs.to(model_device)
        padded_labels = padded_labels.float().to(model_device)

        # predict residue-ligand affinity score 
        binding_embedding, affinity_score = model(res_embeddings, ligand_graphs)
        affinity_score = affinity_score.squeeze(-1)

        pred_score = []
        pred_emb = []
        true_label = []
        
        for i, length in enumerate(origin_len):
            pred_score.append(affinity_score[i, :length])  # 실제 길이만 사용
            pred_emb.append(binding_embedding[i, :length]) # 실제 길이만 사용
            true_label.append(padded_labels[i, :length])   # 실제 길이만 사용
        
        
        pred_score = torch.cat(pred_score, dim=0)
        pred_emb = torch.cat(pred_emb, dim=0)
        true_label = torch.cat(true_label, dim=0)

        # 손실 계산
        bce_loss = criterion(pred_score, true_label).mean()
        
        # Contrastive Loss 적용 여부 체크
        if use_contrastive: 
            nt_xent_loss_value = nt_xent_loss(pred_emb, true_label, temperature=0.2)
            total_loss = (lambda_contrastive * nt_xent_loss_value) + (1-lambda_contrastive) * bce_loss
        else:
            total_loss = bce_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

        avg_loss = train_loss / (batch_idx + 1)
        sys.stdout.write(f'\rBatch {batch_idx+1}/{num_batches} - Train Loss: {avg_loss:.4f}')
        sys.stdout.flush()


def evaluate(test_dataloader, model, criterion, model_device, thresholds=np.arange(0.1, 1.0, 0.001)):
    
    model.eval()  # 평가 모드로 전환
    test_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            res_embeddings, origin_len, ligand_graphs, padded_labels = batch
            
            res_embeddings = res_embeddings.to(model_device)
            ligand_graphs = ligand_graphs.to(model_device)
            padded_labels = padded_labels.to(model_device).float()

            # 모델 예측
            binding_embedding, affinity_score = model(res_embeddings, ligand_graphs)
            affinity_score = affinity_score.squeeze(-1)

            pred_score = []
            true_label = []
            
            for i, length in enumerate(origin_len):
                pred_score.append(affinity_score[i, :length])  # 실제 길이만 사용
                true_label.append(padded_labels[i, :length])   # 실제 길이만 사용
            
            pred_score = torch.cat(pred_score, dim=0)
            true_label = torch.cat(true_label, dim=0)
            
            # 손실 계산
            loss = criterion(pred_score, true_label)
            total_loss = loss.mean()
            test_loss += total_loss.item()
            
            pred_score = torch.sigmoid(pred_score)
            
            # 예측 값과 실제 레이블 저장
            all_preds.extend(pred_score.cpu().numpy().flatten())
            all_labels.extend(true_label.cpu().numpy().flatten())

    # 최적의 binding affinity threshold 찾기
    best_threshold = find_best_threshold(all_preds, all_labels, model_device, thresholds)
    
    # 최적의 threshold에 따른 예측 값 생성
    preds = (np.array(all_preds) > best_threshold).astype(float)

    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    specificity = get_specificity(all_labels, preds)
    mcc = matthews_corrcoef(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    return accuracy, precision, recall, specificity, mcc, f1


def cross_validation(dataset, model_class, model_args, criterion, optimizer_class, 
                     optimizer_args, model_device, num_epochs=10, batch_size=32,
                     lambda_contrastive=0.5, use_contrastive=True):

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = model_class(**model_args).to(model_device)
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            num_batches = len(train_loader)
        
            for batch_idx, batch in enumerate(train_loader):
                res_embeddings, origin_len, ligand_graphs, padded_labels = batch
                res_embeddings = res_embeddings.to(model_device)
                ligand_graphs = ligand_graphs.to(model_device)
                padded_labels = padded_labels.float().to(model_device)

                binding_embedding, affinity_score = model(res_embeddings, ligand_graphs)
                affinity_score = affinity_score.squeeze(-1)

                pred_score = []
                pred_emb = []
                true_label = []
                
                for i, length in enumerate(origin_len):
                    pred_score.append(affinity_score[i, :length])
                    pred_emb.append(binding_embedding[i, :length])
                    true_label.append(padded_labels[i, :length])

                pred_score = torch.cat(pred_score, dim=0)
                pred_emb = torch.cat(pred_emb, dim=0)
                true_label = torch.cat(true_label, dim=0)

                bce_loss = criterion(pred_score, true_label).mean()

                if use_contrastive:
                    contrastive = nt_xent_loss(pred_emb, true_label, temperature=0.2)
                    loss = lambda_contrastive * contrastive + (1 - lambda_contrastive) * bce_loss
                else:
                    loss = bce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                avg_loss = train_loss / (batch_idx + 1)
                
                sys.stdout.write(f'\rFold {fold+1}/10 | Epoch {epoch}/{num_epochs} | Batch {batch_idx+1}/{num_batches} - Train Loss: {avg_loss:.4f}')
                sys.stdout.flush()
            
            print()
            
        acc, prec, rec, mcc, f1, _ = evaluate(val_loader, model, criterion, model_device)
        print(f"Fold {fold+1} | ACC: {acc:.4f} | PRE: {prec:.4f} | REC: {rec:.4f} | MCC: {mcc:.4f} | F1: {f1:.4f}")
        fold_results.append((acc, prec, rec, mcc, f1))
    
    fold_results = np.array(fold_results)
    avg_results = fold_results.mean(axis=0)
    print("\nAverage Cross-Validation Scores")
    print(f"ACC: {avg_results[0]:.4f} | PRE: {avg_results[1]:.4f} | REC: {avg_results[2]:.4f} | MCC: {avg_results[3]:.4f} | F1: {avg_results[4]:.4f}")

    return acc, prec, rec, mcc, f1