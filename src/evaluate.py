from utils import *
from models.libre import *

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved model with utils.evaluate()")

    parser.add_argument("--test_csv", required=True, help="test csv path")
    parser.add_argument("--test_emb", required=True, help="embedding file path")
    parser.add_argument("--test_ligand", required=True, help="ligand file path")
    parser.add_argument("--model_path", required=True, help="model checkpoint (.pt/.pth)")

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--use_cnn_lstm", action="store_true")
    parser.add_argument("--use_ligand", action="store_true")
    parser.add_argument("--residue_input_dim", type=int, default=1280)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, sequence, _, _, origin_len, label, embeddings, ligand = load_data(
        args.test_csv,
        args.test_emb,
        args.test_ligand,
    )
    
    dataloader = create_dataloader(
        sequence,
        origin_len,
        label,
        embeddings,
        ligand,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    model = LiBRe(
        use_cnn_lstm=args.use_cnn_lstm,
        use_ligand=args.use_ligand,
        residue_input_dim=args.residue_input_dim,
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    metrics = evaluate(dataloader, model)

    print("\n=== Evaluation===")
    for k, v in metrics.items():
        try:
            print(f"{k:>7}: {float(v):.4f}")
        except Exception:
            print(f"{k:>7}: {v}")

if __name__ == "__main__":
    main()