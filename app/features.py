"""Признаки детектора слабых сигналов. Вся логика написана здесь, без готовых детекторов трендов.

Идея: редкое упоминание становится сигналом, если одновременно
  1) частота растёт быстрее, чем растёт сам поток публикаций (гамма-пуассоновская оценка на малых числах);
  2) тема молодая и ещё не поднялась по «лестнице зрелости» (стандарты, массовые СМИ, энциклопедия);
  3) о ней пишут разные независимые источники, а не один вендор;
  4) есть деньги и код (ранние раунды, новые репозитории), а не только пресс-релизы.
"""
import datetime as dt
import math
import re

import numpy as np

from . import config

FEATURES = [
    "pub_log_total", "pub_growth_p", "pub_recent_share", "pub_age",
    "news_log_recent", "news_growth_p", "news_saturated", "news_burst", "funding_share", "pr_share",
    "mainstream_share", "domain_diversity", "gh_log_recent", "gh_log_total", "wiki_age", "maturity_share",
]

LABELS = {
    "pub_log_total": "Объём научных публикаций",
    "pub_growth_p": "Ускорение научных публикаций",
    "pub_recent_share": "Доля публикаций за 2 года",
    "pub_age": "Возраст темы в науке",
    "news_log_recent": "Новости за 12 месяцев",
    "news_growth_p": "Рост новостей год к году",
    "news_saturated": "Насыщенный новостной поток",
    "news_burst": "Всплеск новостей (автомат Клейнберга)",
    "funding_share": "Новости о раундах финансирования",
    "pr_share": "Доля пресс-релизов и агрегаторов",
    "mainstream_share": "Доля массовых СМИ",
    "domain_diversity": "Независимость источников",
    "gh_log_recent": "Новые репозитории за год",
    "gh_log_total": "Репозитории всего",
    "wiki_age": "Возраст статьи в Википедии",
    "maturity_share": "Упоминания рынка и лидеров",
}

FUNDING_RE = re.compile(r"\b(raise[sd]?|raising|funding|seed|pre-seed|series [a-e]|investment round|backed by|"
                        r"venture|valuation)\b|привлек|раунд|инвестиц|посевн", re.I)
MATURITY_RE = re.compile(r"market size|market share|billion market|magic quadrant|market leader|widely adopted|"
                         r"mainstream|industry standard|de facto standard|mass adoption|лидер(ы)? рынка|"
                         r"объ[её]м рынка|массов\w* внедр|стандарт де-факто", re.I)
NEWS_CAP = 100  # Google News RSS отдаёт не более 100 записей


def gamma_poisson_prob(x_recent: float, e_recent: float, x_base: float, e_base: float, k: float = 1.5,
                       alpha0: float = 0.5, beta0: float = 0.5, n: int = 4000, seed: int = 7) -> float:
    """P(λ_recent > k·λ_base) при λ ~ Gamma(α0 + x, β0 + e). Устойчиво на счётчиках 0–5,
    где «рост на 200%» из 1→3 ничего не значит: широкий апостериор даёт вероятность около 0,5."""
    if e_recent <= 0 or e_base <= 0:
        return 0.5
    rng = np.random.default_rng(seed)
    lr = rng.gamma(alpha0 + x_recent, 1.0 / (beta0 + e_recent), n)
    lb = rng.gamma(alpha0 + x_base, 1.0 / (beta0 + e_base), n)
    return float(np.mean(lr > k * lb))


def monthly_counts(dates: list, today: dt.date, months: int = 24) -> list:
    """Число документов по месяцам за последние `months` месяцев, от старых к новым."""
    counts = [0] * months
    base = today.year * 12 + today.month
    for s in dates:
        if s and len(s) >= 7 and s[:4].isdigit() and s[5:7].isdigit():
            idx = base - (int(s[:4]) * 12 + int(s[5:7]))
            if 0 <= idx < months:
                counts[months - 1 - idx] += 1
    return counts


def burst_states(counts: list, s: float = 2.0, gamma: float = 1.0) -> list:
    """Двухуровневый автомат всплесков в духе Клейнберга (2002) для помесячных счётчиков.

    Состояние 0 — фоновая интенсивность λ0 = средняя по ряду, состояние 1 — λ1 = s·λ0.
    Стоимость наблюдения x в состоянии с интенсивностью λ: −ln Poisson(x; λ). Подъём 0→1 стоит
    γ·ln T, спуск бесплатен. Оптимальную последовательность состояний находит алгоритм Витерби,
    поэтому единичный скачок не считается всплеском, а устойчивый подъём — считается.
    """
    T, total = len(counts), sum(counts)
    if T == 0 or total == 0:
        return [0] * T
    lam = (total / T, s * total / T)
    up = gamma * math.log(T)

    def emit(x, j):
        return lam[j] - x * math.log(lam[j]) + math.lgamma(x + 1)

    cost = [[emit(counts[0], 0), up + emit(counts[0], 1)]]
    back = [[0, 0]]
    for t in range(1, T):
        c0_from0, c0_from1 = cost[-1][0], cost[-1][1]          # переход в 0: бесплатный из любого состояния
        c1_from0, c1_from1 = cost[-1][0] + up, cost[-1][1]     # переход в 1: подъём стоит up
        b0 = 0 if c0_from0 <= c0_from1 else 1
        b1 = 0 if c1_from0 <= c1_from1 else 1
        cost.append([min(c0_from0, c0_from1) + emit(counts[t], 0), min(c1_from0, c1_from1) + emit(counts[t], 1)])
        back.append([b0, b1])
    state = 0 if cost[-1][0] <= cost[-1][1] else 1
    path = [state]
    for t in range(T - 1, 0, -1):
        state = back[t][state]
        path.append(state)
    return path[::-1]


def _hhi_diversity(values: list[str]) -> float:
    values = [v for v in values if v]
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def compute(ev: dict, today: dt.date | None = None) -> dict:
    today = today or config.today()
    y = today.year
    counts = {int(k): int(v) for k, v in (ev.get("pub_counts") or {}).items()}
    totals = {int(k): int(v) for k, v in (ev.get("pub_totals") or {}).items()}

    # --- наука: объём, рост с нормировкой на объём корпуса, возраст
    total_pubs = sum(counts.values())
    recent_years, base_years = [y - 1, y], [y - 4, y - 3, y - 2]
    x_r = sum(counts.get(t, 0) for t in recent_years)
    x_b = sum(counts.get(t, 0) for t in base_years)
    scale = 1e6
    e_r = sum(totals.get(t, 0) for t in recent_years) / scale
    e_b = sum(totals.get(t, 0) for t in base_years) / scale
    if e_r == 0 or e_b == 0:  # нет знаменателя — годы считаются равными по объёму
        e_r, e_b = 2.0, 3.0
    pub_growth_p = gamma_poisson_prob(x_r, e_r, x_b, e_b) if total_pubs else 0.5
    first_years = [t for t, c in counts.items() if c >= 3 and t <= y]
    pub_age = float(y - min(first_years)) if first_years else 0.0
    pub_recent_share = x_r / total_pubs if total_pubs else 0.0

    # --- новости: объём, рост год к году, структура источников
    # Признаки строятся только по англоязычным новостям: русские есть лишь в окне «последний год»
    # (это исказило бы рост и всплеск), а русские поисковые фразы у разных технологий устроены по-разному.
    # Русские новости остаются в выдаче как источники, но в признаки не входят.
    news_recent = list(ev.get("news_recent") or [])
    n_recent, n_prev = len(news_recent), len(ev.get("news_prev") or [])
    news_growth_p = gamma_poisson_prob(n_recent, 1.0, n_prev, 1.0) if (n_recent or n_prev) else 0.5
    saturated = 1.0 if n_recent >= NEWS_CAP else 0.0

    def _get(d, key):
        return d.get(key) if isinstance(d, dict) else getattr(d, key, "")

    titles = [_get(d, "title") or "" for d in news_recent]
    domains = [_get(d, "domain") or "" for d in news_recent]
    trusts = [_get(d, "trust") or "" for d in news_recent]
    types = [_get(d, "type") or "" for d in news_recent]
    n = len(news_recent)
    funding_share = sum(1 for t in titles if FUNDING_RE.search(t)) / n if n else 0.0
    pr_share = sum(1 for t in trusts if t == "D") / n if n else 0.0
    mainstream_share = sum(1 for t in types if t == "массовое СМИ") / n if n else 0.0
    maturity_share = sum(1 for t in titles if MATURITY_RE.search(t)) / n if n else 0.0

    all_news = list(ev.get("news_recent") or []) + list(ev.get("news_prev") or [])
    series = monthly_counts([_get(d, "date") or "" for d in all_news], today)
    burst = float(np.mean(burst_states(series)[-6:])) if sum(series) >= 6 else 0.0

    gh = ev.get("github") or {}
    wiki = ev.get("wiki") or {}
    wiki_age = float(y - wiki["created_year"]) if wiki.get("exists") and wiki.get("created_year") else 0.0

    return {
        "pub_log_total": math.log1p(total_pubs),
        "pub_growth_p": pub_growth_p,
        "pub_recent_share": pub_recent_share,
        "pub_age": pub_age,
        "news_log_recent": math.log1p(n),
        "news_growth_p": news_growth_p,
        "news_saturated": saturated,
        "news_burst": burst,
        "funding_share": funding_share,
        "pr_share": pr_share,
        "mainstream_share": mainstream_share,
        "domain_diversity": _hhi_diversity(domains),
        "gh_log_recent": math.log1p(int(gh.get("recent", 0))),
        "gh_log_total": math.log1p(int(gh.get("total", 0))),
        "wiki_age": wiki_age,
        "maturity_share": maturity_share,
    }


def ladder(f: dict) -> dict:
    """Лестница зрелости: на каких ступенях уже есть подтверждения."""
    return {
        "research": f["pub_log_total"] > math.log1p(0),
        "code": f["gh_log_total"] > math.log1p(0),
        "money": f["funding_share"] > 0,
        "mainstream": f["mainstream_share"] >= 0.15 or f["news_saturated"] >= 1,
        "encyclopedia": f["wiki_age"] >= 3,
    }


def describe(name: str, value: float) -> str:
    """Человекочитаемое значение признака для объяснения."""
    n = lambda v: int(round(math.expm1(v)))  # noqa: E731
    return {
        "pub_log_total": lambda v: f"{n(v)} научных публикаций всего",
        "pub_growth_p": lambda v: f"публикации ускоряются быстрее корпуса с вероятностью {v:.0%}",
        "pub_recent_share": lambda v: f"{v:.0%} публикаций — за последние 2 года",
        "pub_age": lambda v: f"тема в науке {v:.0f} лет" if v else "устойчивого научного следа нет",
        "news_log_recent": lambda v: f"{n(v)} новостей за 12 месяцев",
        "news_growth_p": lambda v: f"рост новостей год к году с вероятностью {v:.0%}",
        "news_saturated": lambda v: "новостной поток насыщен (100+ за год)" if v else "новостной поток не насыщен",
        "news_burst": lambda v: f"{v:.0%} последних 6 месяцев — в режиме всплеска новостей",
        "funding_share": lambda v: f"{v:.0%} новостей — о раундах финансирования",
        "pr_share": lambda v: f"{v:.0%} источников — пресс-релизы и агрегаторы",
        "mainstream_share": lambda v: f"{v:.0%} новостей — массовые СМИ",
        "domain_diversity": lambda v: f"разнообразие источников {v:.2f} из 1",
        "gh_log_recent": lambda v: f"{n(v)} новых репозиториев за год",
        "gh_log_total": lambda v: f"{n(v)} репозиториев всего",
        "wiki_age": lambda v: f"статья в Википедии существует {v:.0f} лет" if v else "статьи в Википедии нет",
        "maturity_share": lambda v: f"{v:.0%} заголовков говорят о рынке и лидерах",
    }[name](value)
