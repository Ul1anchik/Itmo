import numpy as np

import scipy.special
import torch
import torch.special
from torch.autograd import Function
from scipy.ndimage import gaussian_filter1d
import pywt

# Установим dtype и device для PyTorch (можно изменить при необходимости)
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "Стандартные" значения параметров
m = torch.tensor(1.0, dtype=dtype, device=device) # Масса частицы = масса электрона: 1
sigma = torch.tensor(2590.0, dtype=dtype, device=device) # Продольная ширина волнового пакета в компотоновских длинах волн: 2590 -> 1 nm
pzi = torch.tensor(0.196, dtype=dtype, device=device) # Продольный импульс начального электрона: 0.196 -> 100keV
si = 1 #
li = 1 # Орбитальное квантовое число начального электрона
pzf = torch.tensor(0.19599, dtype=dtype, device=device) # Продольный импульс конечного электрона: 0.19599 -> 99 994.9 eV
sf = 1
lf = 1 # Орбитальное квантовое число конечного электрона

kz = torch.tensor(1.96e-6, dtype=dtype, device=device) # Величина продольного импульса фотона: 1.96e-6 -> 1 eV
kp = torch.tensor(1.96e-7, dtype=dtype, device=device) # Величина поперечного импульса фотона: 1.96e-7 -> 0.1 eV
phi_k = torch.tensor(0.0, dtype=dtype, device=device) # Азимутальный угол импульса фотона: 0

lambda_ = torch.tensor(1.0, dtype=dtype, device=device) # Спиральность фотона: 1
F0 = torch.tensor(7.57e-8, dtype=dtype, device=device) # Сила со стороны электрического поля на заряд: 7.57e-8 -> 100 MeV/m
# t_C = 1.287e-21 s
# lambda_C = 3.86e-13 m
t_in = torch.tensor(1.32e8, dtype=dtype, device=device) # Время начала наблюдения: 2.56e7 t_c = 3.3e-8 s (соответствует расстоянию в 10 мкм при v = c)
t_out = torch.tensor(2.56e10, dtype=dtype, device=device) # Время пролёта ускорителя: 2.56e10 t_c = 3.3e-11 s (соответствует расстоянию в 1 см при v = c)
# H_c = 4.41e9 T
HtoHc = torch.tensor(2.27e-10, dtype=dtype, device=device) # Величина магнитного поля: 2.27e-10 -> 1T
rho_H = 2.0 / torch.sqrt(HtoHc) # Магнитная длина: для H = 1 T магнитная длина 132 744 комптоновских длин волн или 5.12e-8 m = 0.512 nm
# 1 / rho_H соответствует импульсу 7.53e-6 или 3.85 eV
q = torch.tensor(7.297e-3).sqrt().to(dtype=dtype, device=device) # Величина заряда частицы: 7.297e-3 -> q_e
sign = torch.tensor(-1.0, dtype=dtype, device=device) # Знак заряда частицы

kp_max = torch.tensor(1e-4, dtype=dtype, device=device) # Верхний предел интегрирования по величине поперечного импульса фотона: 1e-4 -> 51.1 eV
# kp_max должен быть заметно больше 1 / rho_H
pzf_min = -F0 * t_out # Нижний предел интегрирования по продольному импульсу конечного электрона
# Для F0 = 7.57e-8 = 100 MeV/m и t_out = 2.56e10 = 3.3e-11 s получаем pzf_min = 1.938e3 = 0.99 GeV


def factorial(n):
    if isinstance(n, torch.Tensor):
        return torch.exp(torch.special.gammaln(n + 1))
    else:
        n_tensor = torch.tensor(n, dtype=dtype, device=device)
        return torch.exp(torch.special.gammaln(n_tensor + 1))

# Полиномы Лагерра с положительными верхними индексами
def L_mod_1(n, alpha, x):
    # x — torch.tensor
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 1 + alpha - x
    else:
        L0 = torch.ones_like(x)
        L1 = 1 + alpha - x
        for k in range(2, n + 1):
            Lk = ((2 * k - 1 + alpha - x) * L1 - (k - 1 + alpha) * L0) / k
            L0, L1 = L1, Lk
        return L1

# Полиномы Лагерра с отрицательными верхними индексами
def L_mod(n, alpha, x):
    if alpha > -1:
        return L_mod_1(n,alpha,x)
    else:
        return L_mod_1(n+alpha,-alpha,x)*(-x)**(-alpha)*factorial(n+alpha)/factorial(n)

# Квадрат поперечного импульса
def pp2(s, l, HtoHc, sign):
    l = torch.tensor(l, dtype=dtype, device=device)
    l_abs = torch.abs(l)
    return 2 * HtoHc * (s + l_abs / 2 - sign * l / 2 + 0.5)

# Классическая функция энергии
def En(s, l, pz, t, HtoHc, sign, F0, m):
    return torch.sqrt(m**2 + pp2(s, l, HtoHc, sign) + (pz + F0 * t)**2)

# Классическая продольная траектория
def z(s, l, pz, t, HtoHc, sign, F0, m):
    return (En(s, l, pz, t, HtoHc, sign, F0, m) - En(s, l, pz, 0.0, HtoHc, sign, F0, m)) / F0

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

# Интегральная экспонента
class Exp1Function(Function):
    @staticmethod
    def forward(ctx, input):
        input_np = input.detach().cpu().numpy()
        output_np = scipy.special.exp1(input_np)  # для комплексных значений
        output = torch.from_numpy(np.array(output_np)).to(input.device).type_as(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = -torch.exp(-input) / input
        return grad_output * grad_input

def exp1(x: torch.Tensor) -> torch.Tensor:
    return Exp1Function.apply(x)

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


def wigner_d_matrix_j1_torch(sigma_pol: int, lambda_val: int, beta: torch.Tensor) -> torch.Tensor:

    # Убедимся, что beta является тензором и на правильном устройстве
    if not isinstance(beta, torch.Tensor):
        raise TypeError("beta must be a torch.Tensor")

    # Определение dtype и device из beta
    dtype = beta.dtype
    device = beta.device

    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    sqrt_2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

    # d^1_{m'm}(beta)
    # Используем сложную структуру if/elif/else для охвата всех 9 элементов матрицы.
    # Это позволяет избежать создания большой таблицы поиска и делает логику явной.

    if sigma_pol == 1:
        if lambda_val == 1:
            return (1 + cos_beta) / 2
        elif lambda_val == -1:
            return (1 - cos_beta) / 2
    elif sigma_pol == 0:
        if lambda_val == 1:
            return sin_beta / sqrt_2
        elif lambda_val == -1:
            return -sin_beta / sqrt_2
    elif sigma_pol == -1:
        if lambda_val == 1:
            return (1 - cos_beta) / 2
        elif lambda_val == -1:
            return (1 + cos_beta) / 2

    # Если индексы sigma_pol или lambda_val выходят за пределы [-1, 0, 1]
    raise ValueError(f"Недопустимые индексы для j=1 d-матрицы: sigma_pol={sigma_pol}, lambda_val={lambda_val}. Они должны быть -1, 0 или 1.")

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
