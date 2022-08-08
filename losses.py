import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import math

def infoNCE(h1, h2, temperature=1):
    h1 = F.normalize(h1, dim=-1, p=2)
    h2 = F.normalize(h2, dim=-1, p=2)

    cross_sim = torch.exp(torch.mm(h1, h2.t()) / temperature)
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1)).mean()

def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())

def contrastive_loss_wo_cross_network(h1, h2, z):
    f = lambda x: torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() /
                    (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))


def contrastive_loss_wo_cross_view(h1, h2, z):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, z))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

def CCA_SSG(z1, z2, beta=0.1):
    device = z1.device
    N = z1.size(0)
    D = z1.size(1)
    # batch normalization
    z1_norm = ((z1-z1.mean(0)) / z1.std(0)) / math.sqrt(N)
    z2_norm = ((z2-z2.mean(0)) / z2.std(0)) / math.sqrt(N)

    c1 = torch.mm(z1_norm.T, z1_norm)
    c2 = torch.mm(z2_norm.T, z2_norm)

    iden = torch.eye(D, device=device)
    loss_inv = (z1_norm - z2_norm).pow(2).sum()
    loss_dec_1 = (c1 - iden).pow(2).sum()
    loss_dec_2 = (c2 - iden).pow(2).sum()
    loss_dec = loss_dec_1 + loss_dec_2

    loss = loss_inv+beta*loss_dec
    # print(f"loss_inv: {loss_inv.detach().cpu().item()}, loss_dec: {loss_dec.detach().cpu().item()}")

    return loss

def l_bgrl(q1, q2, y1, y2):
    # x1, x2-> (q1, y2), (q2, y1)
    # [B, D]
    return 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()