

import torch

# Установим dtype и device для PyTorch (можно изменить при необходимости
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
t_in = torch.tensor(2.56e7, dtype=dtype, device=device) # Время начала наблюдения: 2.56e7 t_c = 3.3e-8 s (соответствует расстоянию в 10 мкм при v = c)
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