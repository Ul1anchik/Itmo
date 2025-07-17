
import torch
from .aux_func import En

# Интеграл от энергии от 0 до t, входящий в действие (здесь интегрируем именно от 0, а не от t_in)
def IntEn(s, l, pz, t, HtoHc, sign, F0, m):
    En_t  = En(s, l, pz, t, HtoHc, sign, F0, m)
    En_0  = En(s, l, pz, 0.0, HtoHc, sign, F0, m)
    En00  = En(s, l, 0.0, 0.0, HtoHc, sign, F0, m)

    term1 = (pz + F0 * t) * En_t - pz * En_0

    num   = pz + F0 * t + En_t
    denom = pz + En_0
    # avoid div by 0 or negative log
    eps = 1e-30
    frac = torch.clamp(torch.abs(num / denom), min=eps)

    term2 = En00**2 * torch.log(frac)

    return (term1 + term2) / (2 * F0)