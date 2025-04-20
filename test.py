import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf

from forward_step import ComputeLoss

def eval(model, dataloaders, device, n_gmm, lambda_energy=0.0, lambda_cov=0.0, lambda_kl=0.0, S=20):
    """Testing the DAGMM+VAE model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print("Testing...")

    # we only need compute_params and compute_energy, so KL/reconst weights can be zero
    compute = ComputeLoss(model, lambda_energy, lambda_cov, lambda_kl, device, n_gmm)

    with torch.no_grad():
        # 1) fit GMM params on clean train data
        N_samples = 0
        gamma_sum = 0
        mu_sum    = 0
        cov_sum   = 0

        for x, _ in dataloader_train:
            x = x.float().to(device)
            # unpack to get z and gamma
            _, _, _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = gamma.sum(dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum    += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum   += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            N_samples += x.size(0)

        train_phi = gamma_sum / N_samples
        train_mu  = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # 2) get energies & labels for train set
        energy_train = []
        labels_train = []
        for x, y in dataloader_train:
            x = x.float().to(device)
            _, _, _, _, z, gamma = model(x)
            sample_energy, _ = compute.compute_energy(
                z, gamma,
                phi=train_phi, mu=train_mu, cov=train_cov,
                sample_mean=False
            )
            energy_train.append(sample_energy.cpu())
            labels_train.append(y)
        energy_train = torch.cat(energy_train).numpy()
        labels_train = torch.cat(labels_train).numpy()

        # 3) get energies & labels for test set
        energy_test = []
        labels_test = []
        for x, y in dataloader_test:
            x = x.float().to(device)
            _, _, _, _, z, gamma = model(x)
            sample_energy, _ = compute.compute_energy(
                z, gamma,
                phi=train_phi, mu=train_mu, cov=train_cov,
                sample_mean=False
            )
            energy_test.append(sample_energy.cpu())
            labels_test.append(y)
        energy_test = torch.cat(energy_test).numpy()
        labels_test = torch.cat(labels_test).numpy()

        # combine for thresholding
        scores_total = np.concatenate([energy_train, energy_test], axis=0)
        labels_total = np.concatenate([labels_train, labels_test], axis=0)

    # set threshold at the 100 - S percentile of all scores
    threshold = np.percentile(scores_total, 100 - S)
    preds     = (energy_test > threshold).astype(int)
    gt        = labels_test.astype(int)

    precision, recall, f_score, _ = prf(gt, preds, average='binary')
    roc_auc = roc_auc_score(labels_total, scores_total) * 100

    print(f"Precision : {precision:.4f}, Recall : {recall:.4f}, F-score : {f_score:.4f}")
    print(f"ROC AUC score: {roc_auc:.2f}")
    return labels_total, scores_total