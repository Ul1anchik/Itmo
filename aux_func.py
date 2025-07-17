import numpy as np
import scipy.special
import torch
import torch.special
from torch.autograd import Function

from variables import *

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


