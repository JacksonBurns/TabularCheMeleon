from pathlib import Path

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import Autoencoder

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str)
    parser.add_argument("--latent-dims", type=list, default=[512, 128, 32])
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    # walk through the embeddings directory and load all .pt files, concatenate them into a single tensor
    embeddings = torch.concat([torch.load(f) for f in sorted(Path(args.embeddings_path).glob("*.pt"))])
    dataset = TensorDataset(embeddings)
    # split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = embeddings.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dims=args.latent_dims, lr=args.learning_rate)

    logger = TensorBoardLogger("tb_logs", name="autoencoder", default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1)
    early_stopping_callback = EarlyStopping(monitor="train_loss", patience=args.patience, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    # save only the encoder layers as weights
    torch.save(model.encoder_layers.state_dict(), "chemeleon_ae.pt")
