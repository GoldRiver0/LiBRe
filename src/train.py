import json

from utils import *
from models.libre import *

import torch
import torch.nn as nn

with open('../configs/dataset_config.json', 'r') as f:
    dataset_config = json.load(f)


with open('../configs/model_config.json', 'r') as f:
    model_config = json.load(f)

train_sequence, train_origin_len, train_label, train_embeddings, train_ligand = load_data(
    dataset_config["train_data_path"], 
    dataset_config["train_embedding_path"], 
    dataset_config["train_ligand_path"]
)

coach_sequence, coach_origin_len, coach_label, coach_embeddings, coach_ligand = load_data(
    dataset_config["coach_data_path"], 
    dataset_config["coach_embedding_path"], 
    dataset_config["coach_ligand_path"]
)

holo_sequence, holo_origin_len,  holo_label, holo_embeddings, holo_ligand = load_data(
    dataset_config["holo_data_path"], 
    dataset_config["holo_embedding_path"], 
    dataset_config["holo_ligand_path"]
)

set_seed(42)

train_dataloader = create_dataloader(train_sequence, train_origin_len, train_label, train_embeddings, train_ligand, 32)
coach_dataloader = create_dataloader(coach_sequence, coach_origin_len, coach_label, coach_embeddings, coach_ligand, 16)
holo_dataloader = create_dataloader(holo_sequence, holo_origin_len, holo_label, holo_embeddings, holo_ligand, 16)

BRP_model = BRPredictor(
    use_cnn_lstm=model_config["use_cnn_lstm"],
    use_ligand=model_config["use_ligand"],
    residue_input_dim=model_config["residue_input_dim"]
)

BRP_model.to(model_device)

num_trainable_params = sum(p.numel() for p in BRP_model.parameters() if p.requires_grad)
print(f"Trainable Parameters: {num_trainable_params}")

optimizer = torch.optim.Adam(BRP_model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss(reduction='none')

num_epochs = 200

import time

for epoch in range(1, num_epochs + 1):

        print(f"\nEpoch {epoch}/{num_epochs}")
        
        train(epoch, train_dataloader, BRP_model, criterion, optimizer, model_device)
                
        coach_acc, coach_precision, coach_recall, coach_mcc, coach_f1 = evaluate(coach_dataloader, BRP_model, criterion, model_device)
        holo_acc, holo_precision, holo_recall, holo_mcc, holo_f1 = evaluate(holo_dataloader, BRP_model, criterion, model_device)

        header = f"{'Phase':<12}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'MCC':<12}{'F1 Score':<12}"
        coach_results = f"{'COACH420':<12}{coach_acc:<12.4f}{coach_precision:<12.4f}{coach_recall:<12.4f}{coach_mcc:<12.4f}{coach_f1:<12.4f}"
        holo_results = f"{'HOLO4K':<12}{holo_acc:<12.4f}{holo_precision:<12.4f}{holo_recall:<12.4f}{holo_mcc:<12.4f}{holo_f1:<12.4f}"

        print("=" * 60)
        print(header)
        print("=" * 60)
        print(coach_results)
        print(holo_results)
        print("=" * 60)