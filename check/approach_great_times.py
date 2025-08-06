import torch
import torchquad

from common import *

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Члены суммы по проекциям спина фотона на ось z в S_fi
def TT_wide(si, li, pzi, sf, lf, pzf, kp, y, tin, tout):
    kz = pzi - pzf
    omega = torch.sqrt(kp**2 + kz**2)

    tint1 = Kprime(kp, pzi, pzf, tin, tout) / F0
    denominator = 1j * (omega - kz + 1j * eps)
    tint0 = (2 * torch.exp(1j * (omega - kz) * tin)) / denominator - (
        2 * torch.exp(1j * (omega - kz) * tout)
    ) / denominator

    term1 = Im1(si, li, sf, lf, y) * tint1
    term2 = rho_H * Ffun(si, li, sf, lf, y) * tint0
    term3 = Ip1(si, li, sf, lf, y) * tint1

    return torch.stack([term1, term2, term3])


# Матричный элемент в приближении широкого пакета
def Sfi_wide(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    # Ensure all inputs are tensors and on the correct device
    # Convert scalar inputs to tensors if they aren't already

    si_t = (
        torch.tensor(si, dtype=dtype, device=device)
        if not isinstance(si, torch.Tensor)
        else si.clone().detach()
    )
    li_t = (
        torch.tensor(li, dtype=dtype, device=device)
        if not isinstance(li, torch.Tensor)
        else li.clone().detach()
    )
    sf_t = (
        torch.tensor(sf, dtype=dtype, device=device)
        if not isinstance(sf, torch.Tensor)
        else sf.clone().detach()
    )
    lf_t = (
        torch.tensor(lf, dtype=dtype, device=device)
        if not isinstance(lf, torch.Tensor)
        else lf.clone().detach()
    )
    lambda_t = (
        torch.tensor(lambda_, dtype=dtype, device=device)
        if not isinstance(lambda_, torch.Tensor)
        else lambda_.clone().detach()
    )
    # Pre-calculate common terms for efficiency
    pzi_minus_pzf = pzi - pzf
    kp_sq_plus_pzm_sq = kp**2 + pzi_minus_pzf**2

    sqrt1 = torch.sqrt(
        (2 ** (torch.abs(li_t) + 1))
        / (torch.pi * rho_H**2)
        * factorial(si_t)
        / factorial(si_t + li_t)
    )
    sqrt2 = torch.sqrt(
        (2 ** (torch.abs(lf_t) + 1))
        / (torch.pi * rho_H**2)
        * factorial(sf_t)
        / factorial(sf_t + lf_t)
    )

    photon_norm = 1 / torch.sqrt(2 * torch.sqrt(kp_sq_plus_pzm_sq))
    # Общий префактор
    prefactor = -2 * torch.pi * 1j * photon_norm * sqrt1 * sqrt2 * (q * rho_H) / (2 * m)
    exp_term = torch.exp(1j * (li_t - lf_t) * phi_k)

    # Второй член в разложении разности координат
    z_exp_factor = torch.exp(
        -(
            (
                pzi
                - En(si, li, pzi, 0, HtoHc, sign, F0, m)
                - pzf
                + En(sf, lf, pzf, 0, HtoHc, sign, F0, m)
            )
            ** 2
        )
        / F0**2
        / (4 * sigma**2)
    )

    # Вычисление суммы

    total_sum = torch.zeros_like(kp, dtype=torch.complex128, device=device)
    # Calculate the angle for WignerD
    angle_for_wigner_d = torch.atan2(kp, pzi_minus_pzf)
    # Convert lambda_ to int for wigner_d_matrix_j1_torch
    lambda_int = int(
        lambda_t.item()
    )  # Use .item() to get Python scalar from 0-dim tensor
    for sigma_pol in range(-1, 2):  # sigma_pol = -1, 0, 1
        i_power = (1j) ** (sigma_pol - li_t + lf_t)
        # Get the Wigner D value using the new PyTorch function
        wigner_d_val = wigner_d_matrix_j1_torch(
            sigma_pol, lambda_int, angle_for_wigner_d
        )
        # Get the corresponding element from TT
        # Ensure si, li, sf, lf are passed as Python ints to TT_wide's internal functions
        tt_elements = TT_wide(
            int(si_t.item()),
            int(li_t.item()),
            pzi,
            int(sf_t.item()),
            int(lf_t.item()),
            pzf,
            kp,
            kp * rho_H,
            t_in,
            t_out,
        )
        tt_element = tt_elements[sigma_pol + 1]
        # Add to the sum
        total_sum += i_power * wigner_d_val * tt_element
    result = prefactor * exp_term * total_sum * z_exp_factor
    return result


# ------------------ S_fi -------------------


# Сглаженный матричный элемент
# Только для визуализации - не использовать в сглаженных вероятностях
def Sfi_wide_sm(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    return apply_savgol_filter_to_tensor(
        Sfi_wide(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out), 20
    )


# ------------------ Intensity -------------------


# Спектральная интенсивность: d^2 I / (d pzf d k_perp)
def spec_int_unpol_wide(si, li, pzi, sf, lf, pzf, kp, t_in, t_out):
    energy_factor = torch.sqrt(kp**2 + (pzi - pzf) ** 2)
    measure_factor = (2 * torch.pi) ** -3
    S_elem2 = kp * (
        torch.abs(Sfi_wide(si, li, pzi, sf, lf, pzf, kp, 0, 1, t_in, t_out)) ** 2
        + torch.abs(Sfi_wide(si, li, pzi, sf, lf, pzf, kp, 0, -1, t_in, t_out)) ** 2
    )
    return 2 * torch.pi * measure_factor * S_elem2 * energy_factor


# ------------------ Differential Probability -------------------

# Сглаженная дифференциальная вероятность: d W / d pzf


def dW_dpzf_pol_wide_sm_old(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi) ** -3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        # Используем clone().detach() для входных данных
        x_tensor = (
            x.clone().detach()
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=dtype, device=device)
        )
        return (
            x_tensor
            * torch.abs(
                Sfi_wide_sm(si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out)
            )
            ** 2
        )

    int_result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]]
    )
    return 2 * torch.pi * measure_factor * int_result


def dW_dpzf_pol_wide_sm(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi) ** -3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        # Используем clone().detach() для входных данных
        x_tensor = (
            x.clone().detach()
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=dtype, device=device)
        )
        return apply_savgol_filter_to_tensor(
            x_tensor
            * torch.abs(
                Sfi_wide(si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out)
            )
            ** 2,
            20,
        )

    int_result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]]
    )
    return 2 * torch.pi * measure_factor * int_result


# Просуммированная по поляризациям фотона сглаженная дифференциальная вероятность: d W / d pzf
def dW_dpzf_unpol_wide_sm_old(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_sm_old(
        si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max
    ) + dW_dpzf_pol_wide_sm_old(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)


def dW_dpzf_unpol_wide_sm(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_sm(
        si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max
    ) + dW_dpzf_pol_wide_sm(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)


# Несглаженная дифференциальная вероятность: d W / d pzf
def dW_dpzf_pol_wide(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi) ** -3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        x_tensor = (
            x.clone().detach()
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=dtype, device=device)
        )
        return (
            x_tensor
            * torch.abs(
                Sfi_wide(si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out)
            )
            ** 2
        )

    int_result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]]
    )
    return 2 * torch.pi * measure_factor * int_result


# Просуммированная по поляризациям фотона несглаженная дифференциальная вероятность: d W / d pzf
def dW_dpzf_unpol_wide(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide(
        si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max
    ) + dW_dpzf_pol_wide(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)


# ------------------ Full Probability -------------------


def full_prob(si, li, pzi, sf, lf, t_in, t_out, kp_max, pzf_min):
    gl = torchquad.GaussLegendre()

    def integrand(pzf_batch):  # pzf_batch has shape [N, 1]
        # Распаковать батч:
        pzf_batch = pzf_batch.view(-1)  # теперь [N]

        # Вернуть батч результатов:
        results = torch.stack(
            [
                dW_dpzf_unpol_wide_sm(
                    si, li, pzi, sf, lf, pzf_i.item(), t_in, t_out, kp_max
                )
                for pzf_i in pzf_batch
            ]
        )
        return results  # shape [N]

    result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[pzf_min, pzi]]
    )  # Верхний предел интегрирования - pzi
    return result
