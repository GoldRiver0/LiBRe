from utils2 import *
from models.libre import *

import torch
import torch.nn as nn

args = parse_args()

set_seed(args.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, train_sequence, _, _, train_origin_len, train_label, train_embeddings, train_ligand = load_data(
    args.train_csv,
    args.train_emb,
    args.train_ligand,
)

test_sequence = None
test_origin_len = None
test_label = None
test_embeddings = None
test_ligand = None

if args.test_csv is not None and args.test_emb is not None:
    _, test_sequence, _, _, test_origin_len, test_label, test_embeddings, test_ligand = load_data(
        args.test_csv,
        args.test_emb,
        args.test_ligand,
    )

if args.no_ligand:
    train_ligand = [None] * len(train_ligand)
    if test_sequence is not None:
        test_ligand = [None] * len(test_ligand)

train_bs = 32
eval_bs = 16
lr = 1e-4
epochs = 200

train_dataloader = create_dataloader(
    train_sequence,
    train_origin_len,
    train_label,
    train_embeddings,
    train_ligand,
    batch_size=train_bs,
    shuffle=True,
)

test_dataloader = None
if test_sequence is not None:
    test_dataloader = create_dataloader(
        test_sequence,
        test_origin_len,
        test_label,
        test_embeddings,
        test_ligand,
        batch_size=eval_bs,
        shuffle=False,
    )

model = LiBRe(
    use_cnn_lstm=args.use_cnn_lstm,
    use_ligand=(not args.no_ligand),
    residue_input_dim=args.residue_input_dim,
)

model.to(DEVICE)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters:", num_trainable_params)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss(reduction="none")

for epoch in range(1, epochs + 1):
    train(train_dataloader, model, criterion, optimizer,
          epoch=epoch, epochs=epochs,
          use_contrastive=args.use_contrastive)

    if test_dataloader is not None:
        acc, prec, rec, spec, mcc, f1 = evaluate(test_dataloader, BRP_model)

        header = "{:<10}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}".format(
            "Split", "Accuracy", "Precision", "Recall", "Spec", "MCC", "F1"
        )
        results = "{:<10}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}{:<12.4f}".format(
            "TEST", acc, prec, rec, spec, mcc, f1
        )

        print("=" * 72)
        print(header)
        print("=" * 72)
        print(results)
        print("=" * 72)
