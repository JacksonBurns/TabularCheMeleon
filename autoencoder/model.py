import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dims: list[int],
        lr: float = 1e-3,
    ):
        """
        Args:
            input_dim: Dimensionality of the input feature vector.
            latent_dims: Sequence of hidden/bottleneck dimensions.
                         e.g., [128, 64, 16] creates a 3-layer encoder
                         and a corresponding 3-layer tied decoder.
            lr: Learning rate for Adam optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Full dimension pathway: [input_dim, dim_1, dim_2, ..., latent_dim]
        all_dims = [input_dim] + list(latent_dims)
        self.num_layers = len(latent_dims)

        # Encoder layers: nn.Linear holds weights of shape (out_features, in_features)
        self.encoder_layers = nn.ModuleList([
            nn.Linear(all_dims[i], all_dims[i + 1])
            for i in range(self.num_layers)
        ])

        # Independent decoder biases matching the input dimensions of each encoder step
        self.decoder_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(all_dims[i]))
            for i in range(self.num_layers)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            # Apply non-linearity on intermediate layers; keep bottleneck linear
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        # Traverse encoder layers in reverse order using transposed weights
        for i in reversed(range(self.num_layers)):
            # encoder_layers[i].weight has shape (out_d, in_d).
            # .t() transposes it to (in_d, out_d).
            # F.linear(h, W^T, b) computes h @ (W^T)^T + b = h @ W + b.
            weight_t = self.encoder_layers[i].weight.t()
            bias = self.decoder_biases[i]
            h = F.linear(h, weight_t, bias)

            # Apply non-linearity on intermediate decoder layers
            if i > 0:
                h = F.relu(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def _shared_step(self, batch, step_name: str) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        # Flatten image/spatial inputs if passed as 2D/3D tensors
        if x.dim() > 2:
            x = x.flatten(start_dim=1)

        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
