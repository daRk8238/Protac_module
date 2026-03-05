"""
GNN Training Script
--------------------
Trains the AffinityPredictor GNN on ChEMBL bioactivity data
loaded from the local SQLite database.

Usage:
    python models/train.py
    python models/train.py --epochs 100 --hidden-dim 256 --use-gat
"""

import sys, os, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from database.schema import init_db, get_session, Bioactivity
from models.featurizer import smiles_list_to_graphs
from models.gnn import AffinityPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset(db_path: str = "protac_designer.db"):
    """Load SMILES + pChEMBL pairs from database."""
    engine = init_db(db_path)
    session = get_session(engine)

    records = session.query(Bioactivity).filter(
        Bioactivity.pchembl_value.isnot(None)
    ).all()

    smiles_list, labels = [], []
    seen = set()

    for r in records:
        smi = r.compound.smiles
        val = r.pchembl_value
        if smi and val and smi not in seen:
            smiles_list.append(smi)
            labels.append(float(val))
            seen.add(smi)

    session.close()
    print(f"[Data] Loaded {len(smiles_list)} unique compound-activity pairs")
    return smiles_list, labels


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        fp = batch.fingerprint.view(batch.num_graphs, -1)
        pred = model(batch, fp)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        fp = batch.fingerprint.view(batch.num_graphs, -1)
        pred = model(batch, fp)
        preds.extend(pred.cpu().numpy())
        targets.extend(batch.y.squeeze().cpu().numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    return {
        "rmse": np.sqrt(mean_squared_error(targets, preds)),
        "mae":  mean_absolute_error(targets, preds),
        "r2":   r2_score(targets, preds),
        "preds": preds,
        "targets": targets,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(train_losses, val_metrics, save_dir="models/outputs"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()

    # R² curve
    plt.subplot(1, 2, 2)
    plt.plot([m["r2"] for m in val_metrics], label="Val R²", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R²")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved training curves → {save_dir}/training_curves.png")


def plot_parity(targets, preds, save_dir="models/outputs"):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5, s=20, color="steelblue")
    min_val = min(min(targets), min(preds))
    max_val = max(max(targets), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual pChEMBL")
    plt.ylabel("Predicted pChEMBL")
    plt.title("Predicted vs Actual (Test Set)")
    plt.legend()
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    plt.text(0.05, 0.95, f"R²={r2:.3f}\nRMSE={rmse:.3f}",
             transform=plt.gca().transAxes, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/parity_plot.png", dpi=150)
    plt.close()
    print(f"[Plot] Saved parity plot → {save_dir}/parity_plot.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print(f"[Train] Device: {DEVICE}")
    print(f"[Train] Config: hidden={args.hidden_dim}, layers={args.layers}, "
          f"epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")

    # 1. Load data
    smiles_list, labels = load_dataset(args.db)

    # 2. Featurize
    print("[Train] Converting SMILES to molecular graphs...")
    graphs = smiles_list_to_graphs(smiles_list, labels)
    print(f"[Train] {len(graphs)} valid graphs")

    if len(graphs) < 20:
        print("[ERROR] Not enough data. Run pipeline with more activities first.")
        return

    # 3. Split
    train_graphs, test_graphs = train_test_split(
        graphs, test_size=0.15, random_state=42
    )
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=0.15, random_state=42
    )
    print(f"[Train] Split: {len(train_graphs)} train / {len(val_graphs)} val / {len(test_graphs)} test")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size)

    # 4. Model
    model = AffinityPredictor(
        node_features=9,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        fp_dim=2048,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    # 5. Training loop
    train_losses, val_metrics_history = [], []
    best_val_rmse = float("inf")
    best_epoch = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val RMSE':>10} {'Val R²':>8} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["rmse"])

        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)

        lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {train_loss:>12.4f} {val_metrics['rmse']:>10.4f} "
                  f"{val_metrics['r2']:>8.4f} {lr:>10.2e}")

        # Save best model
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            Path("models/outputs").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "models/outputs/best_model.pt")

    print(f"\n[Train] Best model: epoch {best_epoch}, Val RMSE={best_val_rmse:.4f}")

    # 6. Test set evaluation
    model.load_state_dict(torch.load("models/outputs/best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader)

    print(f"\n{'='*40}")
    print(f"TEST SET RESULTS")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"{'='*40}")

    # 7. Save plots + metrics
    plot_results(train_losses, val_metrics_history)
    plot_parity(test_metrics["targets"], test_metrics["preds"])

    metrics = {
        "test_rmse": test_metrics["rmse"],
        "test_mae":  test_metrics["mae"],
        "test_r2":   test_metrics["r2"],
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_size": len(train_graphs),
        "val_size": len(val_graphs),
        "test_size": len(test_graphs),
    }
    with open("models/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Train] Outputs saved to models/outputs/")
    print(f"  - best_model.pt      (trained weights)")
    print(f"  - training_curves.png")
    print(f"  - parity_plot.png")
    print(f"  - metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",         default="protac_designer.db")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--hidden-dim", type=int,   default=128)
    parser.add_argument("--layers",     type=int,   default=3)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--use-gat",    action="store_true")
    args = parser.parse_args()
    main(args)
