import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Load and preprocess the dataset
class CustomDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.X = self.data[['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12']].values
        self.y = self.data['y'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset('assignment_dataset.csv')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# VAE model
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :self.latent_size], encoded[:, self.latent_size:]
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

# Training the VAE
vae = VAE(input_size=12, latent_size=4)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    for x, y in dataloader:
        x = x.float()
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss = criterion(recon, x) + 0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Generate new synthetic data
latent_samples = torch.randn(1000, vae.latent_size)
synthetic_data = vae.decoder(latent_samples).detach().numpy()