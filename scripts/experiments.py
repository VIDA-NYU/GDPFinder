import os
import torch
from torch.utils.data import DataLoader

from data import get_sample_patches_dataset
from models import AutoEncoder
from train import train_reconstruction
from utils import save_reconstruction_results


def reconstruction_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = os.listdir("../data/output/patches")
    filenames = [os.path.join("../data/output/patches", f) for f in filenames]
    dataset = get_sample_patches_dataset(filenames=filenames)
    dl = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Training AutoEncoder (resnet50) ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset)}")

    model = AutoEncoder(
        latent_dim=64,
        encoder_arch="resnet50",
        encoder_lock_weights=True,
        decoder_latent_dim_channels=128,
        decoder_layers_per_block=[1, 1, 1, 2, 2],
        decoder_enable_bn=False,
    ).to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1000000:d}M"
    )

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_log = train_reconstruction(model, dl, loss, optimizer, device, epochs=20)
    save_reconstruction_results(
        losses_log, dl, model, device, dir="../models/AE_resnet50/"
    )


if __name__ == "__main__":
    reconstruction_experiment()
