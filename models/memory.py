import torch
import torch.nn as nn
from utils import requires_grad


def corrupt_input(x, idx, reverse=True):
    # zero out part of the input
    new_x = x.clone().detach()
    mask = torch.ones_like(x)
    if reverse:
        mask[:, idx:] = 0
    else:
        mask[:, :idx] = 0
    new_x = x * mask
    return new_x.clone().detach()


class Memory(nn.Module):

    def __init__(self, params, device):
        super(Memory, self).__init__()
        self.mem = nn.Linear(params.mem_hid_dim, params.r_dim + params.r2_dim)
        self.b = nn.Parameter(torch.randn(params.mem_hid_dim, device=device, requires_grad=False) * 0.01)
        self.hid_dim = params.mem_hid_dim
        self.step_size = params.mem_step_size
        self.maxiter = params.max_iter
        self.tol = params.tol

        self.device = device

    def predict(self, r):
        return self.mem(r)

    def forward(self, x, mask=None, reverse=False):
        r = self.init_code_(x.size(0))
        opt = torch.optim.Adam([r], lr=self.step_size)
        requires_grad(self.mem.parameters(), False)
        converged = False
        t = 0
        inf_losses = []
        # apply mask if exists (for recall)
        if mask is not None:
            if reverse:
                x = x[:,mask:]
            else:
                x = x[:,:mask]
        while not converged and t < self.maxiter:
            old_r = r.clone()
            x_bar = self.predict(r)
            if mask is not None:
                if reverse:
                    x_bar = x_bar[:,mask:]
                else:
                    x_bar = x_bar[:,:mask]
            loss = torch.pow(x - x_bar, 2).sum(1).mean(0) + torch.pow(r - self.b, 2).sum(1).mean(0)
            loss.backward()
            inf_losses.append(loss.item())
            opt.step()
            opt.zero_grad()
            # check convergence
            with torch.no_grad():
                converged = torch.norm(r - old_r) < self.tol
            t += 1
        requires_grad(self.mem.parameters(), True)
        return r.clone().detach(), inf_losses

    def recall_from_noise(self, x, T=10):
        for _ in range(T):
            # find memory
            r, _ = self.forward(x)
            # reconstruct
            x = self.predict(r).clone().detach()
            # normalize
            x = (x - x.mean(1, keepdim=True))
            x = x / torch.norm(x, dim=1, keepdim=True)
        return x

    def recall_from_partial(self, x, mask, reverse=False):
        # find memory
        r, _ = self.forward(x, mask=mask, reverse=reverse)
        # reconstruct
        x = self.predict(r).clone().detach()
        return x

    def init_code_(self, batch_size):
        return torch.normal(0, 0.025, size=(batch_size, self.hid_dim), requires_grad=True, device=self.device)
        # return torch.zeros(batch_size, self.hid_dim, requires_grad=True, device=self.device)

