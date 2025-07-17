
import numpy as np

from scipy.ndimage import gaussian_filter1d
import torch
import pywt
import torchquad

from .time_and_coord_int import Sfi_wide
from variables import dtype, device

# Функция для сглаживания быстро осциллирующих функций (выдаёт "среднее значение")
def apply_savgol_filter_to_tensor(data_tensor, sigma):
    data_np = data_tensor.clone().detach().cpu().numpy()
    smoothed_np = gaussian_filter1d(data_np, sigma=sigma)
    return torch.as_tensor(smoothed_np, dtype=data_tensor.dtype, device=data_tensor.device)

def smooth_wavelet(data, wavelet='sym4', level=3, sigma=1.0):
    log_data = np.log10(data)

    # Вейвлет-фильтрация
    coeffs = pywt.wavedec(log_data, wavelet, level=level)
    threshold = 0.2 * np.max(np.abs(coeffs[-1]))  # Мягкий порог
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    smoothed_log = pywt.waverec(coeffs, wavelet)

    # Дополнительное Гауссово сглаживание
    smoothed_log = gaussian_filter1d(smoothed_log, sigma=sigma)

    return 10**smoothed_log

# ------------------ S_fi -------------------

# Сглаженный матричный элемент
# Только для визуализации - не использовать в сглаженных вероятностях
def Sfi_wide_sm(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    return apply_savgol_filter_to_tensor(
        Sfi_wide(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out), 20)


# ------------------ Intensity -------------------

# Спектральная интенсивность: d^2 I / (d pzf d k_perp)
def spec_int_unpol_wide(si, li, pzi, sf, lf, pzf, kp, t_in, t_out):
    energy_factor = torch.sqrt(kp**2 + (pzi-pzf)**2)
    measure_factor = (2 * torch.pi)**-3
    S_elem2 = kp * (torch.abs(Sfi_wide(si, li, pzi, sf, lf, pzf, kp, 0, 1, t_in, t_out))**2 +
                   torch.abs(Sfi_wide(si, li, pzi, sf, lf, pzf, kp, 0, -1, t_in, t_out))**2)
    return 2 * torch.pi * measure_factor * S_elem2 * energy_factor


# ------------------ Differential Probability -------------------

# Сглаженная дифференциальная вероятность: d W / d pzf

def dW_dpzf_pol_wide_sm_old(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi)**-3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        # Используем clone().detach() для входных данных
        x_tensor = x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        return x_tensor * torch.abs(Sfi_wide_sm(si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out))**2

    int_result = gl.integrate(integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]])
    return 2 * torch.pi * measure_factor * int_result


def dW_dpzf_pol_wide_sm(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi)**-3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        # Используем clone().detach() для входных данных
        x_tensor = x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        return apply_savgol_filter_to_tensor(x_tensor * torch.abs(Sfi_wide(si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out))**2, 20)

    int_result = gl.integrate(integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]])
    return 2 * torch.pi * measure_factor * int_result

# Просуммированная по поляризациям фотона сглаженная дифференциальная вероятность: d W / d pzf
def dW_dpzf_unpol_wide_sm_old(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_sm_old(si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max) + dW_dpzf_pol_wide_sm_old(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)

def dW_dpzf_unpol_wide_sm(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_sm(si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max) + dW_dpzf_pol_wide_sm(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)

# Несглаженная дифференциальная вероятность: d W / d pzf
def dW_dpzf_pol_wide(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi)**-3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        x_tensor = x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        return x_tensor * torch.abs(Sfi_wide(si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out))**2

    int_result = gl.integrate(integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]])
    return 2 * torch.pi * measure_factor * int_result

# Просуммированная по поляризациям фотона несглаженная дифференциальная вероятность: d W / d pzf
def dW_dpzf_unpol_wide(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide(si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max) + dW_dpzf_pol_wide(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)


# ------------------ Full Probability -------------------

def full_prob(si, li, pzi, sf, lf, t_in, t_out, kp_max, pzf_min):
    gl = torchquad.Gaussian()

    def integrand(pzf_batch):  # pzf_batch has shape [N, 1]
        # Распаковать батч:
        pzf_batch = pzf_batch.view(-1)  # теперь [N]

        # Вернуть батч результатов:
        results = torch.stack([
            dW_dpzf_unpol_wide_sm(si, li, pzi, sf, lf, pzf_i.item(), t_in, t_out, kp_max)
            for pzf_i in pzf_batch
        ])
        return results  # shape [N]

    result = gl.integrate(integrand, dim=1, N=101, integration_domain=[[pzf_min, pzi]]) # Верхний предел интегрирования - pzi
    return result