"""
Trains the two-tower ranker on the train split, evaluates on val, saves the
best checkpoint (by val loss) plus the feature vocab to artifacts/.

Run: python -m app.ranking.train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.data.pipeline import build_and_cache
from app.ranking.dataset import PairDataset
from app.ranking.features import FeatureVocab, feature_dim
from app.ranking.model import TwoTowerModel


def run(csv_path: str, artifacts_dir: str, epochs: int = 8, batch_size: int = 128, lr: float = 1e-3):
    train, val, _test = build_and_cache(csv_path, str(Path(artifacts_dir) / "processed"))
    print(f"train={len(train)} val={len(val)}")

    vocab = FeatureVocab.from_records(train)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    vocab.save(str(Path(artifacts_dir) / "vocab.json"))

    train_ds = PairDataset(train, vocab, pairs_per_user=8, seed=42)
    val_ds = PairDataset(val, vocab, pairs_per_user=8, seed=43)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TwoTowerModel(input_dim=feature_dim(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    ckpt_path = Path(artifacts_dir) / "two_tower.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for user_x, cand_x, label in train_dl:
            opt.zero_grad()
            logits = model(user_x, cand_x)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(label)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_x, cand_x, label in val_dl:
                logits = model(user_x, cand_x)
                val_loss += loss_fn(logits, label).item() * len(label)
        val_loss /= max(len(val_ds), 1)

        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": feature_dim(vocab),
            }, ckpt_path)

    print(f"best val_loss={best_val:.4f}, checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/Assessment_TwitterDataset.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    run(args.csv, args.artifacts, epochs=args.epochs)
