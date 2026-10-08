"""Pre-train the fixed-dim in-context transformer on synthetic CheMeleon tasks.

Each training *episode* is one synthetic task:
  1. sample a variable-size set of examples (rows) from the fixed embedding pool
  2. draw ONE random smooth target function from the prior and score all rows
  3. randomly hide (mask) a fraction of the labels -> those are the "queries"
  4. the model predicts the hidden labels from the visible context in one pass

Over millions of diverse episodes the transformer learns to infer arbitrary
smooth structure from a handful of labeled examples. No column-permutation,
no per-task training.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import FixedDimICLLightning, ModelConfig, FixedDimICLTransformer
from prior import SyntheticPrior


class EpisodicICLDataset(Dataset):
    """On-the-fly synthetic ICL episodes over a fixed embedding pool."""

    def __init__(
        self,
        pool: torch.Tensor,
        prior: SyntheticPrior,
        min_ctx: int = 16,
        max_ctx: int = 256,
        min_query: int = 8,
        max_query: int = 32,
        mask_frac_range: tuple[float, float] = (0.25, 0.75),
        num_episodes: int = 20_000,
        standardise_pool: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.pool = pool.float()
        if standardise_pool:
            mu = self.pool.mean(0)
            sd = self.pool.std(0).clamp_min(1e-6)
            self.pool = (self.pool - mu) / sd
        self.prior = prior
        self.min_ctx = min_ctx
        self.max_ctx = max_ctx
        self.min_query = min_query
        self.max_query = max_query
        self.mask_frac_range = mask_frac_range
        self.num_episodes = num_episodes
        self.rng = torch.Generator().manual_seed(seed)
        self.n = len(self.pool)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict:
        g = self.rng
        # variable context window length (log-uniform)
        n_total = int(round(2 ** torch.rand(1, generator=g).item() *
                            (math.log2(self.max_ctx + self.max_query) -
                             math.log2(self.min_ctx + self.min_query))
                            + math.log2(self.min_ctx + self.min_query)))
        n_total = int(min(max(n_total, self.min_ctx + self.min_query),
                          self.max_ctx + self.max_query))

        idxs = torch.randperm(self.n, generator=g, device=self.pool.device)[:n_total]
        x = self.pool[idxs]                              # (n_total, D)
        y = self.prior(x)                                # (n_total, 1) standardised

        # how many queries to hide
        n_query = int(torch.randint(self.min_query, self.max_query + 1, (1,), generator=g).item())
        n_query = min(n_query, n_total - self.min_ctx)
        n_query = max(n_query, 1)
        qidx = torch.randperm(n_total, generator=g, device=x.device)[:n_query]
        mask = torch.zeros(n_total, dtype=torch.bool, device=x.device)
        mask[qidx] = True

        return {"x": x.cpu(), "y": y.cpu(), "mask": mask}


def collate(batch):
    """Pad a list of variable-length episodes to a common window length.

    Each episode is one context window (context+queries). We right-pad to the
    max length in the batch and build a key_padding_mask so the transformer
    ignores padding tokens.
    """
    B = len(batch)
    D = batch[0]["x"].size(-1)
    max_len = max(b["x"].size(0) for b in batch)

    x = batch[0]["x"].new_zeros(B, max_len, D)
    y = batch[0]["y"].new_zeros(B, max_len, 1)
    mask = batch[0]["mask"].new_zeros(B, max_len, dtype=torch.bool)
    kpm = batch[0]["mask"].new_zeros(B, max_len, dtype=torch.bool)  # True = padding

    for i, b in enumerate(batch):
        L = b["x"].size(0)
        x[i, :L] = b["x"]
        y[i, :L] = b["y"]
        mask[i, :L] = b["mask"]
        kpm[i, L:] = True

    return {"x": x, "y": y, "mask": mask, "key_padding_mask": kpm}


class EpisodicDataModule(pl.LightningDataModule):
    def __init__(self, pool, prior, batch_episodes=4, num_workers=0, **ds_kwargs) -> None:
        super().__init__()
        self.pool = pool
        self.prior = prior
        self.batch_episodes = batch_episodes
        self.num_workers = num_workers
        self.ds_kwargs = ds_kwargs

    def _make(self, train: bool) -> EpisodicICLDataset:
        kw = dict(self.ds_kwargs)
        kw.pop("train_episodes", None)
        kw.pop("val_episodes", None)
        if train:
            kw["num_episodes"] = self.ds_kwargs.get("train_episodes", 20_000)
            kw["seed"] = self.ds_kwargs.get("seed", 0)
        else:
            kw["num_episodes"] = self.ds_kwargs.get("val_episodes", 2048)
            kw["seed"] = self.ds_kwargs.get("seed", 1_000_003)
        return EpisodicICLDataset(self.pool, self.prior, **kw)

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = self._make(True)
        self.val_ds = self._make(False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_episodes, collate_fn=collate,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_episodes, collate_fn=collate,
                          num_workers=self.num_workers, pin_memory=True)


def save_model(lm: FixedDimICLLightning, path: Path) -> None:
    torch.save({"cfg": lm.cfg.to_dict(), "state_dict": lm.model.state_dict()}, path)


def load_model(path: Path, device: str | torch.device = "cpu") -> FixedDimICLTransformer:
    ck = torch.load(path, map_location=device)
    cfg = ModelConfig(**ck["cfg"])
    m = FixedDimICLTransformer(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, default="data/pool_full.pt")
    ap.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=1536)
    ap.add_argument("--max-context", type=int, default=320)
    ap.add_argument("--min-ctx", type=int, default=16)
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--max-query", type=int, default=32)
    ap.add_argument("--batch-episodes", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--train-episodes", type=int, default=20_000)
    ap.add_argument("--val-episodes", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--precision", type=str, default="bf16-mixed")
    ap.add_argument("--log-dir", type=str, default="icl_logs")
    ap.add_argument("--save-ckpt", type=str, default="icl_logs/fixeddim_icl.pt")
    ap.add_argument("--embed-dim", type=int, default=2048)
    args = ap.parse_args()

    pool = torch.load(args.pool, map_location="cpu", weights_only=True)
    print(f"pool shape {tuple(pool.shape)}")

    prior = SyntheticPrior(embed_dim=pool.shape[1])

    cfg = ModelConfig(
        embed_dim=args.embed_dim, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, dim_feedforward=args.dim_ff, dropout=args.dropout,
        max_context=args.max_context, task=args.task,
        learning_rate=args.lr, weight_decay=args.weight_decay,
    )
    lm = FixedDimICLLightning(cfg)

    dm = EpisodicDataModule(
        pool=pool, prior=prior, batch_episodes=args.batch_episodes,
        num_workers=args.num_workers,
        train_episodes=args.train_episodes, val_episodes=args.val_episodes,
        min_ctx=args.min_ctx, max_ctx=args.max_ctx,
        max_query=args.max_query, standardise_pool=True,
    )

    logger = TensorBoardLogger(save_dir=args.log_dir, name="fixeddim_icl", default_hp_metric=False)
    ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1,
                              filename="best-{epoch:02d}-{val/loss:.4f}")
    es = EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto", devices=1, precision=args.precision,
        logger=logger, callbacks=[ckpt_cb, es], log_every_n_steps=50,
        gradient_clip_val=1.0, num_sanity_val_steps=20,
    )
    trainer.fit(lm, dm)

    Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    save_model(lm, Path(args.save_ckpt))
    print(f"saved model -> {args.save_ckpt}")
