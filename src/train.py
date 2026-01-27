from utils import *
from models.libre import *

import os
import copy
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

if not args.use_ligand:
    train_ligand = [None] * len(train_ligand)
    if test_sequence is not None:
        test_ligand = [None] * len(test_ligand)

train_bs = 16
eval_bs = 8
val_ratio = 0.2
lr = 1e-4
epochs = 100

train_dataloader, val_dataloader = make_train_val_loaders(
    train_sequence,
    train_origin_len,
    train_label,
    train_embeddings,
    train_ligand,
    train_bs=train_bs,
    eval_bs=eval_bs,
    val_ratio=val_ratio,
    seed=args.seed,
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
    use_ligand=(args.use_ligand),
    residue_input_dim=args.residue_input_dim,
)

model.to(DEVICE)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters:", num_trainable_params)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss(reduction="none")

best_path = "checkpoints/best_model.pt"
os.makedirs(os.path.dirname(best_path), exist_ok=True)

earlystop = 20      
min_epochs = 20 
min_delta = 1e-4

best_f2 = float("-inf")
best_epoch = 0
patience = 0

print()

for epoch in range(1, epochs + 1):
    print_split_line()

    train(
        train_dataloader, model, criterion, optimizer,
        epoch=epoch, epochs=epochs,
        use_contrastive=args.use_contrastive,
        log_every=10,
    )
    print()

    print_eval_header()

    val_m = evaluate(val_dataloader, model, threshold=0.6)
    print_eval_row("[VAL]", val_m)

    if test_dataloader is not None:
        test_m = evaluate(test_dataloader, model, threshold=0.6)
        print_eval_row("[TEST]", test_m)

    if epoch < min_epochs:
        print(f"  skip best/earlystop ({epoch}/{min_epochs})")
        print_split_line()
        print()
        continue

    is_best = val_m["F2"] > (best_f2 + min_delta)

    if is_best:
        best_f2 = val_m["F2"]
        best_epoch = epoch
        patience = 0
        torch.save(model.state_dict(), best_path)
    else:
        patience += 1

    print(f"BEST: epoch={best_epoch}, val_f2={best_f2:.4f}")
    print_split_line()
    print()

    if patience >= earlystop:
        print(f"Early stopping triggered. Best VAL F2={best_f2:.4f} @ epoch {best_epoch}.")
        break

model.load_state_dict(torch.load(best_path, map_location=DEVICE))

if test_dataloader is not None:
    final_test_m = evaluate(test_dataloader, model, threshold=0.6)
    print("Final(best) TEST")
    print_eval_header()
    print_eval_row("[TEST]", final_test_m)