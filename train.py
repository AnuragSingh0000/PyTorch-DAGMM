import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import DAGMM
from forward_step import ComputeLoss
from utils.utils import weights_init_normal
from test import eval

class TrainerDAGMM:
    """Trainer class for DAGMM + VAE."""
    def __init__(self, args, data, device):
        self.args         = args
        self.train_loader, self.test_loader = data
        self.device       = device

    def train(self):
        """Training the DAGMM+VAE model"""
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # now includes lambda_kl
        self.compute = ComputeLoss(
            self.model,
            self.args.lambda_energy,
            self.args.lambda_cov,
            self.args.lambda_kl,
            self.device,
            self.args.n_gmm
        )

        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            for x, _ in (self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()

                # unpack mu, logvar, z_c, x_hat, z, gamma
                mu, logvar, z_c, x_hat, z, gamma = self.model(x)

                # clamp logvar to [-10, 10] to keep KL stable
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)

                # compute combined DAGMM+VAE loss
                loss = self.compute.forward(x, mu, logvar, x_hat, z, gamma)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            if (epoch % 10 == 0):
                print(f"Training DAGMM+VAE... Epoch: {epoch}, Loss: {avg_loss:.4f}")
                labels, scores = eval(
                    self.model,
                    (self.train_loader, self.test_loader),
                    self.device,
                    self.args.n_gmm,
                    self.args.lambda_energy,
                    self.args.lambda_cov,
                    self.args.lambda_kl
                )

