import torch
import torchquad

from common import *

# Убедитесь, что все ваши вспомогательные функции (En, z, IntEn, Im1, Ffun, Ip1)
# также написаны на PyTorch и могут принимать на вход тензоры.
# Это КРИТИЧЕСКИ ВАЖНО для работы кода.


def TT_wide_Texact(si, li, pzi, sf, lf, pzf, kp, y, tin, tout):
    # Преобразуем скалярные si, li, sf, lf в int для передачи в функции,
    # которые могут ожидать обычные числа (хотя лучше, если они тоже работают с тензорами).
    si, li, sf, lf = int(si), int(li), int(sf), int(lf)

    # Вычисляем константы один раз
    kz = pzi - pzf
    omega = torch.sqrt(kp**2 + kz**2)

    gl = torchquad.GaussLegendre()

    # --- Создаем ЕДИНУЮ векторизованную подынтегральную функцию ---
    def unified_integrand(t_batch, mode):
        # t_batch имеет форму [N, 1], преобразуем в [N] для удобства
        t = t_batch.squeeze(-1)

        # --- Векторизованные вычисления! Нет циклов! Все операции - torch.* ---
        En_i = En(si, li, pzi, t, HtoHc, sign, F0, m)
        En_f = En(sf, lf, pzf, t, HtoHc, sign, F0, m)
        z_i = z(si, li, pzi, t, HtoHc, sign, F0, m)
        z_f = z(sf, lf, pzf, t, HtoHc, sign, F0, m)

        # Общие части для обоих интегралов
        # Добавляем eps для численной стабильности, чтобы избежать деления на ноль
        eps = 1e-30
        prefactor_common = 1.0 / torch.sqrt(torch.clamp(En_i * En_f, min=eps))

        z_diff_sq = (z_i - z_f) ** 2
        exp1 = torch.exp(-z_diff_sq / (4 * sigma**2))

        int_en_diff = IntEn(si, li, pzi, t, HtoHc, sign, F0, m) - IntEn(
            sf, lf, pzf, t, HtoHc, sign, F0, m
        )
        z_sum = z_i + z_f
        phase = omega * t - int_en_diff - z_sum * (pzf - pzi + kz) / 2.0
        exp2 = torch.exp(1j * phase)

        # Различающиеся части
        if mode == "integrand1":
            # Для tint1_Texact у нас просто 1.0 в качестве множителя
            # (но мы оставим его для общности)
            numerator = torch.ones_like(t, dtype=torch.complex128)
        elif mode == "integrand0":
            # Для tint0_Texact
            numerator = -pzi - pzf - 2 * F0 * t + 1j * (z_i - z_f) / sigma**2
        else:
            raise ValueError("Unknown mode")

        return numerator * prefactor_common * exp1 * exp2

    # --- Вычисляем интегралы ---
    # Используем lambda-функции, чтобы передать параметр 'mode' в наш унифицированный интегранд

    # Интеграл для tint1
    tint1_Texact = gl.integrate(
        lambda t: unified_integrand(t, mode="integrand1"),
        dim=1,
        N=101,
        integration_domain=[[tin, tout]],
    )

    # Интеграл для tint0
    tint0_Texact = gl.integrate(
        lambda t: unified_integrand(t, mode="integrand0"),
        dim=1,
        N=101,
        integration_domain=[[tin, tout]],
    )

    # Собираем результат так же, как и раньше
    term1 = Im1(si, li, sf, lf, y) * tint1_Texact
    term2 = rho_H * Ffun(si, li, sf, lf, y) * tint0_Texact
    term3 = (
        Ip1(si, li, sf, lf, y) * tint1_Texact
    )  # В вашем коде тут tint1, предполагаю, что это правильно

    return torch.stack([term1, term2, term3])


def Sfi_wide_Texact(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    # Преобразуем входные данные в тензоры, если они еще не тензоры
    # Ваш способ в целом рабочий, но можно упростить до torch.as_tensor
    si_t = torch.as_tensor(si, dtype=dtype, device=device)
    li_t = torch.as_tensor(li, dtype=dtype, device=device)
    sf_t = torch.as_tensor(sf, dtype=dtype, device=device)
    lf_t = torch.as_tensor(lf, dtype=dtype, device=device)
    lambda_t = torch.as_tensor(lambda_, dtype=dtype, device=device)

    # Pre-calculate common terms for efficiency
    pzi_minus_pzf = pzi - pzf
    kp_sq_plus_pzm_sq = kp**2 + pzi_minus_pzf**2

    # Вынесем преобразование в int за цикл для эффективности
    si_int, li_int = si_t.item(), li_t.item()
    sf_int, lf_int = sf_t.item(), lf_t.item()
    lambda_int = int(lambda_t.item())

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
    prefactor = (
        -2 * torch.pi * 1j * photon_norm * sqrt1 * sqrt2 * (q * rho_H) / (2 * m)
    )  # заменил q на e
    exp_term = torch.exp(1j * (li_t - lf_t) * phi_k)

    # Этот член z_exp_factor был частью приближения. При точном интегрировании он
    # уже неявно учтен внутри интеграла. Его нужно убрать из префактора.
    # z_exp_factor = torch.exp(-(pzi - En(...) - pzf + En(...))**2 / F0**2 / (4 * sigma**2)) # <-- УДАЛИТЬ

    total_sum = torch.zeros_like(kp, dtype=torch.complex128, device=device)
    angle_for_wigner_d = torch.atan2(kp, pzi_minus_pzf)

    for sigma_pol in range(-1, 2):  # sigma_pol = -1, 0, 1
        i_power = (1j) ** (
            torch.tensor(sigma_pol, dtype=li_t.dtype, device=device) - li_t + lf_t
        )

        wigner_d_val = wigner_d_matrix_j1_torch(
            sigma_pol, lambda_int, angle_for_wigner_d
        )

        # --- ГЛАВНОЕ ИЗМЕНЕНИЕ ЗДЕСЬ ---
        # Вызываем новую функцию с численным интегрированием
        tt_elements = TT_wide_Texact(
            si_int, li_int, pzi, sf_int, lf_int, pzf, kp, kp * rho_H, t_in, t_out
        )

        tt_element = tt_elements[sigma_pol + 1]

        total_sum += i_power * wigner_d_val * tt_element

    # Умножаем на префактор без z_exp_factor
    result = prefactor * exp_term * total_sum
    return result


# ------------------------- ВЕРОЯТНОСТИ С ТОЧНЫМ ЧИСЛЕННЫМ ИНТЕГРИРОВАНИЕМ ПО ВРЕМЕНИ ------------------------
# ------------------ S_fi -------------------


# Сглаженный матричный элемент
# Только для визуализации - не использовать в сглаженных вероятностях
def Sfi_wide_sm_Texact(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out):
    return apply_savgol_filter_to_tensor(
        Sfi_wide_Texact(si, li, pzi, sf, lf, pzf, kp, phi_k, lambda_, t_in, t_out), 20
    )


# ------------------ Intensity -------------------


# Спектральная интенсивность: d^2 I / (d pzf d k_perp)
def spec_int_unpol_wide_Texact(si, li, pzi, sf, lf, pzf, kp, t_in, t_out):
    energy_factor = torch.sqrt(kp**2 + (pzi - pzf) ** 2)
    measure_factor = (2 * torch.pi) ** -3
    S_elem2 = kp * (
        torch.abs(Sfi_wide_Texact(si, li, pzi, sf, lf, pzf, kp, 0, 1, t_in, t_out)) ** 2
        + torch.abs(Sfi_wide_Texact(si, li, pzi, sf, lf, pzf, kp, 0, -1, t_in, t_out))
        ** 2
    )
    return 2 * torch.pi * measure_factor * S_elem2 * energy_factor


# ------------------ Differential Probability -------------------

# Сглаженная дифференциальная вероятность: d W / d pzf


def dW_dpzf_pol_wide_sm_old_Texact(
    si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max
):
    measure_factor = (2 * torch.pi) ** -3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        x_tensor = x.squeeze(-1)
        return (
            x_tensor
            * torch.abs(
                Sfi_wide_sm_Texact(
                    si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out
                )
            )
            ** 2
        )

    # ### ИСПРАВЛЕНО: Извлекаем только результат интеграла, игнорируя ошибку
    int_result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]]
    )[0]
    return 2 * torch.pi * measure_factor * int_result


def dW_dpzf_pol_wide_sm_Texact(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi) ** -3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        x_tensor = x.squeeze(-1)
        # ВАЖНО: pzf здесь - это ОДНО число Python, а не тензор.
        sfi_result_sq = (
            torch.abs(
                Sfi_wide_Texact(
                    si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out
                )
            )
            ** 2
        )
        return apply_savgol_filter_to_tensor(x_tensor * sfi_result_sq, 20)

    # ### ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ:
    # Эта функция всегда вызывается с одним pzf, поэтому gl.integrate вернет ТЕНЗОР-СКАЛЯР.
    # Мы НЕ индексируем его. Он просто возвращается как результат.
    int_result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]]
    )
    return 2 * torch.pi * measure_factor * int_result


def dW_dpzf_unpol_wide_sm_old_Texact(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_sm_old_Texact(
        si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max
    ) + dW_dpzf_pol_wide_sm_old_Texact(
        si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max
    )


def dW_dpzf_unpol_wide_sm_Texact(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_sm_Texact(
        si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max
    ) + dW_dpzf_pol_wide_sm_Texact(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)


# Несглаженная дифференциальная вероятность: d W / d pzf


def dW_dpzf_pol_wide_Texact(si, li, pzi, sf, lf, pzf, lambda_, t_in, t_out, kp_max):
    measure_factor = (2 * torch.pi) ** -3
    gl = torchquad.GaussLegendre()

    def integrand(x):
        x_tensor = x.squeeze(-1)
        return (
            x_tensor
            * torch.abs(
                Sfi_wide_Texact(
                    si, li, pzi, sf, lf, pzf, x_tensor, 0, lambda_, t_in, t_out
                )
            )
            ** 2
        )

    # ### ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: То же самое здесь. Просто возвращаем тензор.
    int_result = gl.integrate(
        integrand, dim=1, N=101, integration_domain=[[1e-10, float(kp_max)]]
    )
    return 2 * torch.pi * measure_factor * int_result


def dW_dpzf_unpol_wide_Texact(si, li, pzi, sf, lf, pzf, t_in, t_out, kp_max):
    return dW_dpzf_pol_wide_Texact(
        si, li, pzi, sf, lf, pzf, 1, t_in, t_out, kp_max
    ) + dW_dpzf_pol_wide_Texact(si, li, pzi, sf, lf, pzf, -1, t_in, t_out, kp_max)


# ------------------ Full Probability -------------------


def full_prob_Texact(si, li, pzi, sf, lf, t_in, t_out, kp_max, pzf_min):
    gl = torchquad.GaussLegendre()

    # ### ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: Возвращаемся к циклу - это ЕДИНСТВЕННЫЙ верный путь.
    def integrand(pzf_batch):
        # pzf_batch - это тензор формы [N, 1].
        # Мы итерируемся по нему, чтобы вычислить dW/dpzf для каждой точки pzf.

        results = []
        for pzf_scalar_tensor in pzf_batch:
            # Превращаем тензор с одним элементом в простое число Python.
            pzf_val = pzf_scalar_tensor.item()

            # Вычисляем dW/dpzf для ОДНОГО значения pzf_val.
            # Эта функция вернет 0-мерный тензор (скаляр).
            dW_dpzf_val = dW_dpzf_unpol_wide_sm_Texact(
                si, li, pzi, sf, lf, pzf_val, t_in, t_out, kp_max
            )
            results.append(dW_dpzf_val)

        # Собираем список 0-мерных тензоров в один 1-мерный тензор (вектор).
        return torch.stack(results)

    # ### ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ:
    # Этот финальный вызов возвращает тензор-скаляр, который и является ответом.
    # Мы НЕ индексируем его.
    result = gl.integrate(integrand, dim=1, N=101, integration_domain=[[pzf_min, pzi]])
    return result
