# schreyer_ae.py - adversarial autoencoder baseline
# in: feature matrix (n_entries, 618), fitted on train rows only
# out: one anomaly score per entry
# ---------------------------------------------------------------
#
# Refactor of the implementation published with the original paper.
# The architecture and the learning rates are unchanged
# (encoder and decoder 1e-3, discriminator and generator 1e-5),
# so this is a fair reproduction and not a reinterpretation.
# Only surrounding code was rewritten to fit the fit/score interface the other baselines use

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.adam import Adam
from typing import List, Optional
import numpy as np
import os, io
import urllib.request


# define encoder class
class Encoder(nn.Module):

    # define class constructor
    def __init__(self, input_size, hidden_size):

        # call super class constructor
        super(Encoder, self).__init__()

        # specify first layer - in 618, out 256
        self.map_L1 = nn.Linear(input_size, hidden_size[0], bias=True)  # init linearity
        nn.init.xavier_uniform_(self.map_L1.weight)  # init weights according to [9]
        nn.init.constant_(self.map_L1.bias, 0.0)  # constant initialization of the bias
        self.map_R1 = nn.LeakyReLU(
            negative_slope=0.4, inplace=True
        )  # add non-linearity according to [10]

        # specify second layer - in 256, out 64
        self.map_L2 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        nn.init.xavier_uniform_(self.map_L2.weight)
        nn.init.constant_(self.map_L2.bias, 0.0)
        self.map_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify third layer - in 64, out 16
        self.map_L3 = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.map_L3.weight)
        nn.init.constant_(self.map_L3.bias, 0.0)
        self.map_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify fourth layer - in 16, out 4
        self.map_L4 = nn.Linear(hidden_size[2], hidden_size[3], bias=True)
        nn.init.xavier_uniform_(self.map_L4.weight)
        nn.init.constant_(self.map_L4.bias, 0.0)
        self.map_R4 = torch.nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify fifth layer - in 4, out 2
        self.map_L5 = nn.Linear(hidden_size[3], hidden_size[4], bias=True)
        nn.init.xavier_uniform_(self.map_L5.weight)
        nn.init.constant_(self.map_L5.bias, 0.0)
        self.map_R5 = torch.nn.LeakyReLU(negative_slope=0.4, inplace=True)

    # define forward pass
    def forward(self, x):

        # run forward pass through the network
        x = self.map_R1(self.map_L1(x))
        x = self.map_R2(self.map_L2(x))
        x = self.map_R3(self.map_L3(x))
        x = self.map_R4(self.map_L4(x))
        x = self.map_R5(self.map_L5(x))

        # return result
        return x


# define decoder class
class Decoder(nn.Module):

    # define class constructor
    def __init__(self, output_size, hidden_size):

        # call super class constructor
        super(Decoder, self).__init__()

        # specify first layer - in 2, out 4
        self.map_L1 = nn.Linear(
            hidden_size[0], hidden_size[1], bias=True
        )  # init linearity
        nn.init.xavier_uniform_(self.map_L1.weight)  # init weights according to [9]
        nn.init.constant_(self.map_L1.bias, 0.0)  # constant initialization of the bias
        self.map_R1 = nn.LeakyReLU(
            negative_slope=0.4, inplace=True
        )  # add non-linearity according to [10]

        # specify second layer - in 4, out 16
        self.map_L2 = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.map_L2.weight)
        nn.init.constant_(self.map_L2.bias, 0.0)
        self.map_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify third layer - in 16, out 64
        self.map_L3 = nn.Linear(hidden_size[2], hidden_size[3], bias=True)
        nn.init.xavier_uniform_(self.map_L3.weight)
        nn.init.constant_(self.map_L3.bias, 0.0)
        self.map_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify fourth layer - in 64, out 256
        self.map_L4 = nn.Linear(hidden_size[3], hidden_size[4], bias=True)
        nn.init.xavier_uniform_(self.map_L4.weight)
        nn.init.constant_(self.map_L4.bias, 0.0)
        self.map_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify fifth layer - in 256, out 618
        self.map_L5 = nn.Linear(hidden_size[4], output_size, bias=True)
        nn.init.xavier_uniform_(self.map_L5.weight)
        nn.init.constant_(self.map_L5.bias, 0.0)
        self.map_S5 = torch.nn.Sigmoid()

    # define forward pass
    def forward(self, x):

        # run forward pass through the network
        x = self.map_R1(self.map_L1(x))
        x = self.map_R2(self.map_L2(x))
        x = self.map_R3(self.map_L3(x))
        x = self.map_R4(self.map_L4(x))
        x = self.map_S5(self.map_L5(x))

        # return result
        return x


# define discriminator class
class Discriminator(nn.Module):

    # define class constructor
    def __init__(self, input_size, hidden_size, output_size):

        # call super class constructor
        super(Discriminator, self).__init__()

        # specify first layer - in 2, out 256
        self.map_L1 = nn.Linear(input_size, hidden_size[0], bias=True)  # init linearity
        nn.init.xavier_uniform_(self.map_L1.weight)  # init weights according to [9]
        nn.init.constant_(self.map_L1.bias, 0.0)  # constant initialization of the bias
        self.map_R1 = nn.LeakyReLU(
            negative_slope=0.4, inplace=True
        )  # add non-linearity according to [10]

        # specify second layer - in 256, out 16
        self.map_L2 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        nn.init.xavier_uniform_(self.map_L2.weight)
        nn.init.constant_(self.map_L2.bias, 0.0)
        self.map_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify third layer - in 16, out 4
        self.map_L3 = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.map_L3.weight)
        nn.init.constant_(self.map_L3.bias, 0.0)
        self.map_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify fourth layer - in 4, out 2
        self.map_L4 = nn.Linear(hidden_size[2], output_size, bias=True)
        nn.init.xavier_uniform_(self.map_L4.weight)
        nn.init.constant_(self.map_L4.bias, 0.0)
        self.map_S4 = torch.nn.Sigmoid()

    # define forward pass
    def forward(self, x):

        # run forward pass through the network
        x = self.map_R1(self.map_L1(x))
        x = self.map_R2(self.map_L2(x))
        x = self.map_R3(self.map_L3(x))
        x = self.map_S4(self.map_L4(x))

        # return result
        return x


class SchreyerAEBaseline:

    TAU = 5
    RADIUS = 0.8
    SIGMA = 0.01
    LATENT_DIM = 2

    def __init__(
        self,
        input_dim: int = 206,
        num_categorical: int = 204,
        hidden_size: Optional[List[int]] = None,
        device: str = "cpu",
        alpha: float = 0.4,
        mini_batch_size: int = 128,
    ):
        self.input_dim = input_dim
        self.num_categorical = num_categorical
        self.device = device
        self.alpha = alpha
        self.mini_batch_size = mini_batch_size

        if hidden_size is None:
            hidden_size = [64, 32, 16, 8, self.LATENT_DIM]

        self.hidden_size = hidden_size

        encoder_hidden = hidden_size
        decoder_hidden = list(reversed(hidden_size))

        self.encoder = Encoder(input_dim, encoder_hidden).to(device)
        self.decoder = Decoder(input_dim, decoder_hidden).to(device)
        self.discriminator = Discriminator(
            input_size=self.LATENT_DIM, hidden_size=[256, 16, 4], output_size=1
        ).to(device)

        self.mu_gauss = self._build_gaussian_prior()
        self.is_fitted = False

    def _build_gaussian_prior(self) -> np.ndarray:
        x_centroid = (
            self.RADIUS * np.sin(np.linspace(0, 2 * np.pi, self.TAU, endpoint=False))
            + 1
        ) / 2
        y_centroid = (
            self.RADIUS * np.cos(np.linspace(0, 2 * np.pi, self.TAU, endpoint=False))
            + 1
        ) / 2
        mu_gauss = np.vstack([x_centroid, y_centroid]).T
        return mu_gauss

    def _sample_prior(self, n: int) -> np.ndarray:
        samples_per_gauss = n // self.TAU + 1
        samples = []
        for mu in self.mu_gauss:
            s = np.random.normal(
                mu, self.SIGMA, size=(samples_per_gauss, self.LATENT_DIM)
            )
            samples.append(s)
        all_samples = np.vstack(samples)
        idx = np.random.choice(all_samples.shape[0], size=n, replace=False)
        return all_samples[idx]

    def fit(
        self,
        x_train: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        lr_disc: float = 1e-5,
    ):
        """
        Train the autoencoder and the discriminator alternately

        Next to the reconstruction loss, the discriminator pushes the latent
        space towards a chosen Gaussian prior.

        That is what makes a distance to the prior a meaningful anomaly signal in this model,
        and it is exactly the constraint the GNN of this thesis does not have
        """
        reconstruction_criterion_categorical = nn.BCELoss(reduction="mean").to(
            self.device
        )
        reconstruction_criterion_numeric = nn.MSELoss(reduction="mean").to(self.device)
        discriminator_criterion = nn.BCELoss(reduction="mean").to(self.device)

        # define optimizers
        encoder_optimizer = Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = Adam(self.decoder.parameters(), lr=lr)
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=lr_disc)
        generator_optimizer = Adam(self.encoder.parameters(), lr=lr_disc)

        # convert pre-processed data to pytorch tensor
        torch_dataset = TensorDataset(
            torch.from_numpy(x_train.astype(np.float32)).float()
        )
        dataloader = DataLoader(
            torch_dataset, batch_size=self.mini_batch_size, shuffle=True, num_workers=0
        )
        # note: we set num_workers to zero to retrieve deterministic results

        print(f"Training: {epochs} epochs, {len(dataloader)} batches/epoch")

        for epoch in range(epochs):
            epoch_rec_loss = 0.0
            epoch_dis_loss = 0.0
            mini_batch_count = 0

            for mini_batch_data in dataloader:
                mini_batch_count += 1

                # convert mini batch to torch variable
                mini_batch_torch = mini_batch_data[0].to(
                    device=self.device, dtype=torch.float32
                )

                # reset the networks gradients
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.discriminator.zero_grad()

                # =================== reconstruction phase =====================

                # run autoencoder encoding - decoding
                z_sample = self.encoder(mini_batch_torch)
                mini_batch_reconstruction = self.decoder(z_sample)

                # split input data to numerical and categorical part
                batch_cat = mini_batch_torch[:, : self.num_categorical]
                batch_num = mini_batch_torch[:, self.num_categorical :]

                # split reconstruction to numerical and categorical part
                rec_batch_cat = mini_batch_reconstruction[:, : self.num_categorical]
                rec_batch_num = mini_batch_reconstruction[:, self.num_categorical :]

                # compute reconstruction losses
                rec_error_cat = reconstruction_criterion_categorical(
                    input=rec_batch_cat, target=batch_cat
                )
                rec_error_num = reconstruction_criterion_numeric(
                    input=rec_batch_num, target=batch_num
                )
                reconstruction_loss = rec_error_cat + rec_error_num

                reconstruction_loss.backward()
                epoch_rec_loss += reconstruction_loss.item()

                decoder_optimizer.step()
                encoder_optimizer.step()

                # =================== regularization phase =====================
                # =================== discriminator training ===================

                self.discriminator.eval()

                # generate target latent space data from the Gaussian prior
                z_target_batch_np = self._sample_prior(mini_batch_torch.shape[0])
                z_target_batch = torch.tensor(
                    z_target_batch_np, dtype=torch.float32, device=self.device
                )

                # determine mini batch sample generated by the encoder -> fake gaussian sample
                z_fake_gauss = self.encoder(mini_batch_torch)

                # determine discriminator classification of both samples
                d_real_gauss = self.discriminator(
                    z_target_batch
                )  # real sampled gaussian
                d_fake_gauss = self.discriminator(
                    z_fake_gauss.detach()
                )  # fake created gaussian

                # determine discriminator classification target variables
                d_real_gauss_target = torch.ones(
                    d_real_gauss.shape, dtype=torch.float32, device=self.device
                )
                d_fake_gauss_target = torch.zeros(
                    d_fake_gauss.shape, dtype=torch.float32, device=self.device
                )

                # determine individual discrimination losses
                discriminator_loss_real = discriminator_criterion(
                    input=d_real_gauss, target=d_real_gauss_target
                )
                discriminator_loss_fake = discriminator_criterion(
                    input=d_fake_gauss, target=d_fake_gauss_target
                )
                discriminator_loss = discriminator_loss_fake + discriminator_loss_real

                discriminator_loss.backward()
                epoch_dis_loss += discriminator_loss.item()
                discriminator_optimizer.step()

                # =================== generator training =======================

                self.discriminator.train()
                self.encoder.zero_grad()

                z_fake_gauss = self.encoder(mini_batch_torch)
                d_fake_gauss = self.discriminator(z_fake_gauss)
                generator_target = torch.ones(
                    d_fake_gauss.shape, dtype=torch.float32, device=self.device
                )
                generator_loss = discriminator_criterion(
                    input=d_fake_gauss, target=generator_target
                )

                generator_loss.backward()
                generator_optimizer.step()

            if (epoch + 1) % 10 == 0:
                avg_rec = epoch_rec_loss / mini_batch_count
                avg_dis = epoch_dis_loss / mini_batch_count
                print(
                    f"  Epoch [{epoch+1:3d}/{epochs}] | rec_loss: {avg_rec:.4f} | dis_loss: {avg_dis:.4f}"
                )

        self.encoder.eval()
        self.decoder.eval()
        self._is_fitted = True
        return self

    def score(self, x_eval: np.ndarray) -> np.ndarray:
        """Anomaly score, the reconstruction error combined with the distance to
        the prior, weighted by alpha = 0.8 as in the original paper
        """
        assert self._is_fitted, "Call fit() before score()"

        # define the optimization criterion / loss function (eval mode, reduction='none')
        reconstruction_criterion_categorical_eval = nn.BCELoss(reduction="none").to(
            self.device
        )
        reconstruction_criterion_numeric_eval = nn.MSELoss(reduction="none").to(
            self.device
        )

        # set networks in evaluation mode (don't apply dropout)
        self.encoder.eval()
        self.decoder.eval()

        # convert pre-processed data to pytorch tensor
        torch_dataset = TensorDataset(
            torch.from_numpy(x_eval.astype(np.float32)).float()
        )
        dataloader_eval = DataLoader(
            torch_dataset, batch_size=self.mini_batch_size, shuffle=False, num_workers=0
        )

        z_enc_batches: list[torch.Tensor] = []
        rec_error_batches: list[torch.Tensor] = []

        with torch.no_grad():
            for enc_transactions_batch in dataloader_eval:

                # push mini batch to selected device
                enc_transactions_batch = enc_transactions_batch[0].to(
                    device=self.device, dtype=torch.float32
                )

                # determine latent space representation of all transactions
                z_enc_transactions_batch = self.encoder(enc_transactions_batch)

                # reconstruct input samples
                reconstruction_batch = self.decoder(z_enc_transactions_batch)

                # split input transactions into categorical and numeric parts
                input_cat_all = enc_transactions_batch[:, : self.num_categorical]
                input_num_all = enc_transactions_batch[:, self.num_categorical :]

                # split reconstruction into categorical and numeric parts
                rec_cat_all = reconstruction_batch[:, : self.num_categorical]
                rec_num_all = reconstruction_batch[:, self.num_categorical :]

                # compute rec error
                rec_error_cat_all = reconstruction_criterion_categorical_eval(
                    input=rec_cat_all, target=input_cat_all
                ).mean(dim=1)
                rec_error_num_all = reconstruction_criterion_numeric_eval(
                    input=rec_num_all, target=input_num_all
                ).mean(dim=1)

                # combine categorical and numerical errors
                rec_error_all_batch = rec_error_cat_all + rec_error_num_all

                z_enc_batches.append(z_enc_transactions_batch)
                rec_error_batches.append(rec_error_all_batch)

        z_enc_transactions_all = torch.cat(z_enc_batches, dim=0).cpu().detach().numpy()
        rec_error_all = torch.cat(rec_error_batches, dim=0).cpu().detach().numpy()

        def compute_euclid_distance(x, y):
            euclidean_distance = np.sqrt(np.sum((x - y) ** 2, axis=1))
            return euclidean_distance

        distances = np.apply_along_axis(
            func1d=compute_euclid_distance,
            axis=1,
            arr=z_enc_transactions_all,
            y=self.mu_gauss,
        )

        mode_divergence = np.min(distances, axis=1)

        cluster_ids = np.argmin(distances, axis=1)

        rec_error_all_scaled = np.asarray(rec_error_all, dtype=np.float64)
        for cluster_id in np.unique(cluster_ids).tolist():
            mask = cluster_ids == cluster_id
            rec_error_range = np.ptp(rec_error_all[mask])
            if rec_error_range == 0:
                rec_error_all_scaled[mask] = 0.0
            else:
                rec_error_all_scaled[mask] = (
                    rec_error_all[mask] - rec_error_all[mask].min()
                ) / rec_error_range

        mode_divergence_all_scaled = np.asarray(mode_divergence, dtype=np.float64)
        for cluster_id in np.unique(cluster_ids).tolist():
            mask = cluster_ids == cluster_id
            mode_divergence_range = np.ptp(mode_divergence[mask])
            if mode_divergence_range == 0:
                mode_divergence_all_scaled[mask] = 0.0
            else:
                mode_divergence_all_scaled[mask] = (
                    mode_divergence[mask] - mode_divergence[mask].min()
                ) / mode_divergence_range

        anomaly_score = (
            self.alpha * rec_error_all_scaled
            + (1.0 - self.alpha) * mode_divergence_all_scaled
        )

        return anomaly_score
