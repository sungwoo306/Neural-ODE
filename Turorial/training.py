#%%
import numpy as np
import torch

dataset_numpy = np.load("./test_dataset.npz")


class AFMSpectroscopyDataset(torch.utils.data.Dataset):
    def __init__(self, t, u0, z, u):
        super().__init__()
        self.t = torch.tensor(t)
        self.u0 = torch.tensor(u0)
        self.z = torch.tensor(z)
        self.u = torch.tensor(u)

    def __len__(self) -> int:
        return self.u0.size(0)

    def __getitem__(self, idx) -> int:
        sample = {
            "t": self.t[idx],
            "u0": self.u0[idx],
            "args": self.z[idx],
            "u": self.u[idx],
        }
        return sample


dataset = AFMSpectroscopyDataset(**dataset_numpy)
# %%
from models import FullyConnectedNetwork


class ForcedHarmonicOscillator(torch.nn.Module):
    def __init__(self, Q, k, A0, f0, fd, F_int):
        super().__init__()
        self.Q = torch.tensor(Q)
        self.k = torch.tensor(k)
        self.A0 = torch.tensor(A0)
        self.w0 = torch.tensor(2 * torch.pi * f0)
        self.wd = torch.tensor(2 * torch.pi * fd)
        self.F_int = F_int
        self.F0 = self.k * self.A0

    def forward(
        self, t: torch.Tensor, u: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        dxi = u[:, 1].view(-1, 1)
        ddxi = (
            -u[:, 1].view(-1, 1) / self.Q
            - u[:, 0].view(-1, 1)
            + torch.cos(self.wd * t.view(-1, 1) / self.w0) / self.Q
            + self.F_int((u[:, 0] + z).view(-1, 1) * self.A0) / self.F0
        )
        return torch.cat((dxi, ddxi), dim=-1)


# %%
surrogate = torch.jit.script(
    FullyConnectedNetwork([1, 32, 32, 1], torch.nn.functional.gelu)
)
surrogate.output_layer.weight.data.fill_(0.0)
surrogate.output_layer.bias.data.fill_(0.0)
# surrogate = lambda z: 0
eom = ForcedHarmonicOscillator(500, 25, 2e-9, 160000, 160000, surrogate)

# %%
eom
# %%
from odesolve import odesolve

sol = odesolve(
    eom,
    dataset.u0.to("cuda"),
    dataset.t.to("cuda"),
    dataset.z.to("cuda"),
    method="tsit5",
)
# %%
sol.ys.size()
# %%
t_sol = sol.ts.detach().cpu()
y_sol = sol.ys.detach().cpu()
# %%
import matplotlib.pyplot as plt

n_plots = 5
time_ind = slice(0, 1000)
fig, axes = plt.subplots(
    n_plots, 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True
)
plot_inds = torch.flip(
    torch.linspace(0, t_sol.size(0) - 1, n_plots, dtype=int), dims=[0]
)
for ind, ax in zip(plot_inds, axes):
    ax.plot(t_sol[ind], dataset.u[ind, :, 0], linewidth=0.8, label="Data")
    ax.plot(
        t_sol[ind],
        y_sol[ind, :, 0],
        linewidth=0.8,
        label="prediction",
    )

axes[0].legend()
axes[0].set_title(r"Normalized deflection $\tilde{\xi}(t)$")
axes[-1].set_xlabel("Normalized time $\omega_0 t$")
# %%
import pytorch_lightning as pl
from neuralode import NeuralODE
from pytorch_lightning.callbacks import ModelCheckpoint

NODE = NeuralODE(eom, lr=1e-3)
if __name__ == "__main__":
    pl.seed_everything(10)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        num_workers=8,
        pin_memory=True,
    )
    checkpoint_every_n_epochs = 200
    trainer = pl.Trainer(
        max_epochs=4000,
        callbacks=[
            ModelCheckpoint(
                monitor="train_loss",
                mode="min",
                save_top_k=1,
                filename="best_{epoch}-{step}",
            ),
            ModelCheckpoint(
                monitor="train_loss",
                every_n_epochs=checkpoint_every_n_epochs,
                save_top_k=-1,
                filename="{epoch}-{step}",
            ),
        ],
        log_every_n_steps=1,
        auto_select_gpus=True,
        deterministic="warn",
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(NODE, train_dataloader)
# %%
