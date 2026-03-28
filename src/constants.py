import torch

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = torch.tensor(1.0, dtype=dtype, device=device)
sigma = torch.tensor(2590.0, dtype=dtype, device=device)
pzi = torch.tensor(0.196, dtype=dtype, device=device)
si = 1
li = 1
pzf = torch.tensor(0.19599, dtype=dtype, device=device)
sf = 1
lf = 1

kz = torch.tensor(1.96e-6, dtype=dtype, device=device)
kp = torch.tensor(1.96e-7, dtype=dtype, device=device)
phi_k = torch.tensor(0.0, dtype=dtype, device=device)
lambda_ = torch.tensor(1.0, dtype=dtype, device=device)
F0 = torch.tensor(7.57e-8, dtype=dtype, device=device)

t_in = torch.tensor(2.56e7, dtype=dtype, device=device)
t_out = torch.tensor(2.56e10, dtype=dtype, device=device)

HtoHc = torch.tensor(2.27e-10, dtype=dtype, device=device)
rho_H = 2.0 / torch.sqrt(HtoHc)

q = torch.tensor(7.297e-3).sqrt().to(dtype=dtype, device=device)
sign = torch.tensor(-1.0, dtype=dtype, device=device)

kp_max = torch.tensor(1e-4, dtype=dtype, device=device)
pzf_min = -F0 * t_out

eps = torch.tensor(1e-30)
