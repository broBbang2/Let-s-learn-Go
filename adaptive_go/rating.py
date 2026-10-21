"""Glicko-2 레이팅 (Glickman, "Example of the Glicko-2 system", 2013).

Elo와 달리 레이팅 편차(RD)를 같이 들고 있어서, 신규 유저는 RD가 커서 빨리 움직이고
판이 쌓일수록 RD가 줄어 안정된다. 대국 결과 기반 실력 추정의 바탕.
"""

import math
from dataclasses import dataclass

SCALE = 173.7178  # Glicko → Glicko-2 내부 척도 변환 상수
TAU = 0.5  # 변동성(volatility)이 얼마나 빨리 바뀔 수 있는지. 논문 권장 0.3~1.2
EPSILON = 1e-6


@dataclass
class Rating:
    rating: float = 1500.0
    rd: float = 350.0
    volatility: float = 0.06


def _g(phi):
    return 1 / math.sqrt(1 + 3 * phi * phi / (math.pi * math.pi))


def _expected(mu, mu_j, phi_j):
    return 1 / (1 + math.exp(-_g(phi_j) * (mu - mu_j)))


def update(player, results, tau=TAU):
    """한 레이팅 기간의 결과로 새 Rating을 계산한다.

    results: [(opponent Rating, score), ...]  score는 승 1, 무 0.5, 패 0
    """
    mu = (player.rating - 1500) / SCALE
    phi = player.rd / SCALE
    sigma = player.volatility

    if not results:
        # 대국이 없으면 RD만 커진다 (오래 안 두면 불확실해짐)
        return Rating(player.rating, min(350.0, math.sqrt(phi * phi + sigma * sigma) * SCALE), sigma)

    v_inv, delta_sum = 0.0, 0.0
    for opp, score in results:
        mu_j = (opp.rating - 1500) / SCALE
        phi_j = opp.rd / SCALE
        g, e = _g(phi_j), _expected(mu, mu_j, phi_j)
        v_inv += g * g * e * (1 - e)
        delta_sum += g * (score - e)
    v = 1 / v_inv
    delta = v * delta_sum

    sigma_new = _new_volatility(phi, sigma, v, delta, tau)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1 / math.sqrt(1 / (phi_star * phi_star) + 1 / v)
    mu_new = mu + phi_new * phi_new * delta_sum
    return Rating(mu_new * SCALE + 1500, phi_new * SCALE, sigma_new)


def _new_volatility(phi, sigma, v, delta, tau):
    """논문 Step 5: Illinois 방법으로 f(x)=0의 근을 찾는다."""
    a = math.log(sigma * sigma)

    def f(x):
        ex = math.exp(x)
        return ex * (delta * delta - phi * phi - v - ex) / (2 * (phi * phi + v + ex) ** 2) - (x - a) / (tau * tau)

    big_a = a
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        big_b = a - k * tau

    f_a, f_b = f(big_a), f(big_b)
    while abs(big_b - big_a) > EPSILON:
        big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(big_c)
        if f_c * f_b <= 0:
            big_a, f_a = big_b, f_b
        else:
            f_a /= 2
        big_b, f_b = big_c, f_c
    return math.exp(big_a / 2)
