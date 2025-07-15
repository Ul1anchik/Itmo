import torch
import numpy as np
from src.constants import F0, t_in, t_out, rho_H, eps, m, q, sign
from src.utils import Ffun, factorial, exp1, En, wigner_d_matrix_j1_torch
from src.constants import dtype, device

# Приближённый интеграл по времени до бесконечности
def K(kp, pzi, pzf, t_in):
    kz = pzi - pzf
    dpz = pzi + pzf
    phase = -1j / (2 * F0) * (torch.sqrt(kp ** 2 + kz ** 2) - kz) * dpz
    arg = -1j * (torch.sqrt(kp ** 2 + kz ** 2) - kz + 1j * 1e-40) * (t_in + dpz / (2 * F0))
    return torch.exp(phase) * exp1(arg)

# Приближённый интеграл по времени от t_in до t_out
def Kprime(kp, pzi, pzf, t_in, t_out):
    return K(kp, pzi, pzf, t_in) - K(kp, pzi, pzf, t_out)

# Интегралы Ip1 и Im1
def Ip1(si, li, sf, lf, y):
    term1 = 2 * Ffun(si, li, sf, lf + 1, y)
    term2 = (si + li) * Ffun(si, li - 1, sf, lf, y)
    return (-1j) * torch.sqrt(torch.tensor(2.0, device=device)) * (term1 + term2)

def Im1(si, li, sf, lf, y):
    term1 = 2 * Ffun(si, li + 1, sf, lf, y)
    term2 = (sf + lf) * Ffun(si, li, sf, lf - 1, y)
    return (-1j) * torch.sqrt(torch.tensor(2.0, device=device)) * (term1 + term2)

# Члены суммы TT_wide
def TT_wide(si, li, pzi, sf, lf, pzf, kp, y, tin, tout):
    kz = pzi - pzf
    omega = torch.sqrt(kp**2 + kz**2)

    tint1 = Kprime(kp, pzi, pzf, tin, tout) / F0
    denominator = 1j * (omega - kz + 1j * eps)
    tint0 = (2 * torch.exp(1j * (omega - kz) * tin)) / denominator - \
            (2 * torch.exp(1j * (omega - kz) * tout)) / denominator

    term1 = Im1(si, li, sf, lf, y) * tint1
    term2 = rho_H * Ffun(si, li, sf, lf, y) * tint0
    term3 = Ip1(si, li, sf, lf, y) * tint1

    return torch.stack([term1, term2, term3])

# Основная амплитуда в приближении широкого пакета
def Sfi_wide(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    si_t = torch.tensor(si, dtype=dtype, device=device) if not isinstance(si, torch.Tensor) else si.clone().detach()
    li_t = torch.tensor(li, dtype=dtype, device=device) if not isinstance(li, torch.Tensor) else li.clone().detach()
    sf_t = torch.tensor(sf, dtype=dtype, device=device) if not isinstance(sf, torch.Tensor) else sf.clone().detach()
    lf_t = torch.tensor(lf, dtype=dtype, device=device) if not isinstance(lf, torch.Tensor) else lf.clone().detach()
    lambda_t = torch.tensor(lambda_, dtype=dtype, device=device) if not isinstance(lambda_, torch.Tensor) else lambda_.clone().detach()

    pzi_minus_pzf = pzi - pzf
    kp_sq_plus_pzm_sq = kp**2 + pzi_minus_pzf**2

    sqrt1 = torch.sqrt((2**(torch.abs(li_t) + 1)) / (torch.pi * rho_H**2) * factorial(si_t) / factorial(si_t + li_t))
    sqrt2 = torch.sqrt((2**(torch.abs(lf_t) + 1)) / (torch.pi * rho_H**2) * factorial(sf_t) / factorial(sf_t + lf_t))

    photon_norm = 1 / torch.sqrt(2 * torch.sqrt(kp_sq_plus_pzm_sq))
    prefactor = -2 * torch.pi * 1j * photon_norm * sqrt1 * sqrt2 * (q * rho_H) / (2 * m)
    exp_term = torch.exp(1j * (li_t - lf_t) * phi_k)

    z_exp_factor = torch.exp(-(pzi - En(si, li, pzi, 0, HtoHc, sign, F0, m) - pzf + En(sf, lf, pzf, 0, HtoHc, sign, F0, m))**2 / F0**2 / (4 * sigma**2))

    total_sum = torch.zeros_like(kp, dtype=torch.complex128, device=device)
    angle_for_wigner_d = torch.atan2(kp, pzi_minus_pzf)
    lambda_int = int(lambda_t.item())

    for sigma_pol in range(-1, 2):
        i_power = (1j)**(sigma_pol - li_t + lf_t)
        wigner_d_val = wigner_d_matrix_j1_torch(sigma_pol, lambda_int, angle_for_wigner_d)
        tt_elements = TT_wide(int(si_t.item()), int(li_t.item()), pzi, int(sf_t.item()), int(lf_t.item()), pzf, kp, kp * rho_H, t_in, t_out)
        tt_element = tt_elements[sigma_pol + 1]
        total_sum += i_power * wigner_d_val * tt_element

    return prefactor * exp_term * total_sum * z_exp_factor
