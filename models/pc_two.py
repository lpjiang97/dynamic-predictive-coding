import torch.nn as nn
import torch
import torch.nn.functional as F


def soft_thresholding(r, lmda):
    with torch.no_grad():
        rtn = F.relu(F.relu(r - lmda) - F.relu(-r - lmda))
    return rtn.data


class DynPredNet(nn.Module):

    def __init__(self, params, device):
        super(DynPredNet, self).__init__()
        # spatial: p(I | r)
        self.spatial_decoder = nn.Linear(params.r_dim, params.input_dim, bias=False)
        # temporal: p(r_t | r_(t-1), r2)
        self.temporal = nn.Parameter(torch.randn(params.mix_dim, params.r_dim, params.r_dim, requires_grad=True))
        nn.init.xavier_uniform_(self.temporal)
        self.hypernet = nn.Sequential(
            nn.Linear(params.r2_dim, params.hyper_hid_dim),
            nn.LayerNorm(params.hyper_hid_dim),
            nn.ELU(),
            nn.Linear(params.hyper_hid_dim, params.hyper_hid_dim),
            nn.Linear(params.hyper_hid_dim, params.mix_dim),
        )
        # hyperparams
        self.device = device
        self.r_dim = params.r_dim
        self.r2_dim = params.r2_dim
        self.mix_dim = params.mix_dim

        self.lr_r = params.lr_r
        self.lmda_r = params.lmda_r
        self.lr_r2 = params.lr_r2
        self.lmda_r2 = params.lmda_r2
        self.temp_weight = params.temp_weight

        self.max_iter = params.max_iter
        self.tol = params.tol

    def forward(self, X):
        batch_size = X.size(0)
        T = X.size(1)
        r, r2 = self.init_code_(batch_size)
        r = self.inf_first_step(X[:, 0])
        r_first = r.clone().detach()
        spatial_loss = torch.pow(X[:, 0] - self.spatial_decoder(r), 2).view(batch_size, -1).sum(1).mean(0)
        temp_loss = 0
        # img var
        r2_losses = torch.zeros((batch_size, T-1), device=self.device)
        for t in range(1, T):
            r_p = r.clone().detach()
            r, r2, r2_loss = self.inf(X[:, t], r_p, r2.clone().detach())
            r2_losses[:,t-1] = r2_loss
            # learning
            x_hat = self.spatial_decoder(r)
            r_hat = self.temporal_prediction_(r_p, r2)
            # loss
            spatial_loss += torch.pow(X[:, t] - x_hat, 2).view(batch_size, -1).sum(1).mean(0)
            temp_loss += torch.pow(r - r_hat, 2).view(batch_size, -1).sum(1).mean(0)
        return spatial_loss, self.temp_weight * temp_loss, r2_losses, r_first, r2.clone().detach()

    def inf(self, x, r_p, r2):
        batch_size = x.size(0)
        r, _ = self.init_code_(batch_size)
        r2.requires_grad = True
        orig_r2 = r2.clone().detach()
        # fit r
        optim_r = torch.optim.SGD([r], self.lr_r)
        optim_r2 = torch.optim.Adam([r2], self.lr_r2, weight_decay=self.lmda_r2)
        converged = False
        i = 0
        # inference
        while not converged and i < self.max_iter:
            old_r = r.clone().detach()
            old_r2 = r2.clone().detach()
            # prediction
            x_bar = self.spatial_decoder(r)
            r_bar = self.temporal_prediction_(r_p, r2)
            # prediction error
            spatial_loss = torch.pow(x - x_bar, 2).view(batch_size, -1).sum(1).mean(0)
            temporal_loss = torch.pow(r - r_bar, 2).view(batch_size, -1).sum(1).mean(0)
            # update neural activity
            loss = spatial_loss + self.temp_weight * temporal_loss
            #r2_loss.append(temporal_loss.item())
            loss.backward()
            optim_r.step()
            optim_r2.step()
            optim_r.zero_grad()
            optim_r2.zero_grad()
            self.zero_grad()
            # shrinkage
            r.data = soft_thresholding(r, self.lmda_r)
            # convergence
            with torch.no_grad():
                converged = torch.norm(r - old_r) / (torch.norm(old_r) + 1e-16) < self.tol \
                        and torch.norm(r2 - old_r2) / (torch.norm(old_r2) + 1e-16) < self.tol
            i += 1
        # compute the error (prior vs. posterior)
        r2_loss = torch.pow(r - self.temporal_prediction_(r_p, orig_r2).clone().detach(), 2).view(batch_size, -1).sum(1)
        self.converge_warning_(i, "r/r2 did not converge")
        return r.clone().detach(), r2.clone().detach(), r2_loss

    def inf_first_step(self, x):
        batch_size = x.size(0)
        r, _ = self.init_code_(batch_size)
        optim = torch.optim.SGD([r], self.lr_r)
        converged = False
        i = 0
        while not converged and i < self.max_iter:
            old_r = r.clone().detach()
            # prediction
            x_bar = self.spatial_decoder(r)
            # prediction error
            loss = torch.pow(x - x_bar, 2).view(batch_size, -1).sum(1).mean(0)
            # update neural activity
            loss.backward()
            optim.step()
            optim.zero_grad()
            self.zero_grad()
            # shrinkage
            r.data = soft_thresholding(r, self.lmda_r)
            # convergence
            with torch.no_grad():
                #print(torch.norm(r - old_r) / (torch.norm(old_r) + 1e-16))
                converged = torch.norm(r - old_r) / (torch.norm(old_r) + 1e-16) < self.tol
            i += 1
        self.converge_warning_(i, "first step r did not converge")
        return r.clone().detach()

    def temporal_prediction_(self, r, r2):
        batch_size = r.size(0)
        w = self.hypernet(r2)
        V_t = torch.matmul(w, self.temporal.reshape(self.mix_dim, -1)).reshape(batch_size, self.r_dim, self.r_dim)
        r_hat = F.relu(torch.bmm(V_t, r.unsqueeze(2))).squeeze()
        return r_hat

    def init_code_(self, batch_size):
        r = torch.zeros((batch_size, self.r_dim), requires_grad=True, device=self.device)
        r2 = torch.zeros((batch_size, self.r2_dim), requires_grad=True, device=self.device)
        return r, r2

    def zero_grad(self):
        self.spatial_decoder.zero_grad()
        if self.temporal.grad is not None:
            self.temporal.grad.zero_()
        self.hypernet.zero_grad()

    def normalize(self):
        with torch.no_grad():
            self.spatial_decoder.weight.data = F.normalize(self.spatial_decoder.weight.data, dim=0)

    def converge_warning_(self, i, msg):
        if i >= self.max_iter:
            print(msg)

