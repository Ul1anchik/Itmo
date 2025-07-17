
import torch

from variables import *
from aux_func import *

# Интеграл по времени от t_in до бесконечности в приближении F0 * t >> pzi, pzf
def K(kp, pzi, pzf, t_in):
    kz = pzi - pzf
    dpz = pzi + pzf
    phase = -1j / (2 * F0) * (torch.sqrt(kp ** 2 + kz ** 2) - kz) * dpz
    arg = -1j * (torch.sqrt(kp ** 2 + kz ** 2) - kz + 1j * 1e-40) * (t_in + dpz / (2 * F0))
    return torch.exp(phase) * exp1(arg)

# интеграл по времени от t_in до t_out
def Kprime(kp, pzi, pzf, t_in, t_out):
    return K(kp, pzi, pzf, t_in) - K(kp, pzi, pzf, t_out)

# Интеграл по поперечной координате
def Ffun(si, li, sf, lf, y):
    sf_lf_fact = factorial(sf + lf)
    si_fact = factorial(si)

    exponent = 3*(si - sf) + 2*li - lf + 2
    y_power = y**(2*(si - sf) + li - lf)

    y_sq = y**2
    L1 = L_mod(sf + lf, (si - sf + li - lf), y_sq / torch.tensor(8, dtype=dtype, device=device))
    L2 = L_mod(sf, (si - sf), y_sq / torch.tensor(8, dtype=dtype, device=device))

    exp_term = torch.exp(-y_sq / torch.tensor(8, dtype=dtype, device=device))

    return (sf_lf_fact / si_fact) * (1 / 2**exponent) * y_power * L1 * L2 * exp_term

def Ip1(si, li, sf, lf, y):
    term1 = 2 * Ffun(si, li, sf, lf + 1, y)
    term2 = (si + li) * Ffun(si, li - 1, sf, lf, y)
    return (-1j) * torch.sqrt(torch.tensor(2.0, device=device)) * (term1 + term2)

def Im1(si, li, sf, lf, y):
    term1 = 2 * Ffun(si, li + 1, sf, lf, y)
    term2 = (sf + lf) * Ffun(si, li, sf, lf - 1, y)
    return (-1j) * torch.sqrt(torch.tensor(2.0, device=device)) * (term1 + term2)

# Регуляризация
eps = torch.tensor(1e-30)

# Члены суммы по проекциям спина фотона на ось z в S_fi
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


# Матричный элемент в приближении широкого пакета
def Sfi_wide(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    # Ensure all inputs are tensors and on the correct device
    # Convert scalar inputs to tensors if they aren't already

    si_t = torch.tensor(si, dtype=dtype, device=device) if not isinstance(si, torch.Tensor) else si.clone().detach()
    li_t = torch.tensor(li, dtype=dtype, device=device) if not isinstance(li, torch.Tensor) else li.clone().detach()
    sf_t = torch.tensor(sf, dtype=dtype, device=device) if not isinstance(sf, torch.Tensor) else sf.clone().detach()
    lf_t = torch.tensor(lf, dtype=dtype, device=device) if not isinstance(lf, torch.Tensor) else lf.clone().detach()
    lambda_t = torch.tensor(lambda_, dtype=dtype, device=device) if not isinstance(lambda_, torch.Tensor) else lambda_.clone().detach()
    # Pre-calculate common terms for efficiency
    pzi_minus_pzf = pzi - pzf
    kp_sq_plus_pzm_sq = kp**2 + pzi_minus_pzf**2

    sqrt1 = torch.sqrt((2**(torch.abs(li_t) + 1)) / (torch.pi * rho_H**2) * factorial(si_t) / factorial(si_t + li_t))
    sqrt2 = torch.sqrt((2**(torch.abs(lf_t) + 1)) / (torch.pi * rho_H**2) * factorial(sf_t) / factorial(sf_t + lf_t))

    photon_norm = 1 / torch.sqrt(2 * torch.sqrt(kp_sq_plus_pzm_sq))
    # Общий префактор
    prefactor = -2 * torch.pi * 1j * photon_norm * sqrt1 * sqrt2 * (q * rho_H) / (2 * m)
    exp_term = torch.exp(1j * (li_t - lf_t) * phi_k)
    # Вычисление суммы

    total_sum = torch.zeros_like(kp, dtype=torch.complex128, device=device)
    # Calculate the angle for WignerD
    angle_for_wigner_d = torch.atan2(kp, pzi_minus_pzf)
    # Convert lambda_ to int for wigner_d_matrix_j1_torch
    lambda_int = int(lambda_t.item()) # Use .item() to get Python scalar from 0-dim tensor
    for sigma_pol in range(-1, 2):  # sigma_pol = -1, 0, 1
        i_power = (1j)**(sigma_pol - li_t + lf_t)
        # Get the Wigner D value using the new PyTorch function
        wigner_d_val = wigner_d_matrix_j1_torch(sigma_pol, lambda_int, angle_for_wigner_d)
        # Get the corresponding element from TT
        # Ensure si, li, sf, lf are passed as Python ints to TT_wide's internal functions
        tt_elements = TT_wide(int(si_t.item()), int(li_t.item()), pzi, int(sf_t.item()), int(lf_t.item()), pzf, kp, kp * rho_H, t_in, t_out)
        tt_element = tt_elements[sigma_pol + 1]
        # Add to the sum
        total_sum += i_power * wigner_d_val * tt_element
    result = prefactor * exp_term * total_sum
    return result