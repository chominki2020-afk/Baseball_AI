"""
섞어 던지기 (게임이론) — 앱·AI 분석관에서 쓰는 계산

구종별 기대 실점(rv0)은 매치업 모델(R.simulate)에서, 구종마다 '노릴 만한 코스들'(그 구종 상위 25% 코스) 평균.
타자가 노리는 효과: 구종 x를 평소보다 많이 던지면 x의 기대 실점이 β·(구사율 − 평소 구사율)만큼 나빠진다고 가정.
β 기본값은 MLB 투수들의 실제 카운트별 선택에 가장 잘 맞는 값 (build_game.py mix — '역 게임이론').
"""
import os
import pickle

import numpy as np
import pandas as pd

from build_game import expected, game_mix

DEFAULT_BETA = 20.0


def load(path="game_mix.pkl", path2="game_exploit.pkl"):
    g = pickle.load(open(path, "rb")) if os.path.exists(path) else None
    e = pickle.load(open(path2, "rb")) if os.path.exists(path2) else None
    return g, e


def rv0_from_sim(res, top=0.25):
    """R.simulate 결과 → 구종별 (노릴 코스 평균 기대 실점, 그 구종 최선 코스, 평소 구사율)"""
    rows = []
    for t, g in res.groupby("pitch_type"):
        g = g.sort_values("rv100")
        k = max(1, int(len(g) * top))
        best = g.iloc[0]
        rows.append({"pitch_type": t, "rv0": g["rv100"].iloc[:k].mean(), "usage": float(g["usage"].iloc[0]),
                     "tx": best["tx"], "tz": best["tz"], "best_rv": best["rv100"]})
    df = pd.DataFrame(rows)
    df = df[df["usage"] >= 0.03].reset_index(drop=True)
    df["usage"] = df["usage"] / df["usage"].sum()
    return df


def plan(res, beta=DEFAULT_BETA):
    df = rv0_from_sim(res)
    rv0, u_ref = df["rv0"].to_numpy(), df["usage"].to_numpy()
    mix = game_mix(rv0, u_ref, beta)
    greedy = np.zeros(len(df))
    greedy[np.argmin(rv0)] = 1
    df["mix"] = mix
    ev = {"제일 좋은 공만": expected(greedy, rv0, u_ref, beta), "평소 비율 그대로": expected(u_ref, rv0, u_ref, beta),
          "게임이론 추천": expected(mix, rv0, u_ref, beta)}
    return df, ev


# ---------------------------------------------------------------- 미니맥스(무차별 원리) 검증 결과
def load_minimax(path="minimax.pkl"):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


GRP_KO = {"g1": "변화구", "g2": "오프스피드"}


def replicated(mm):
    """두 시즌 모두 95% 구간이 0을 안 걸치고 같은 방향인 카운트×계열"""
    a, b = mm["by_count"][2024].set_index("count"), mm["by_count"][2025].set_index("count")
    out = []
    for c in a.index.intersection(b.index):
        for g in ("g1", "g2"):
            s1 = np.sign(a.loc[c, g]) if (a.loc[c, g + "_lo"] > 0 or a.loc[c, g + "_hi"] < 0) else 0
            s2 = np.sign(b.loc[c, g]) if (b.loc[c, g + "_lo"] > 0 or b.loc[c, g + "_hi"] < 0) else 0
            if s1 != 0 and s1 == s2:
                out.append({"count": c, "group": GRP_KO[g], "v24": a.loc[c, g], "v25": b.loc[c, g],
                            "fb_share": (a.loc[c, "fb_share"] + b.loc[c, "fb_share"]) / 2})
    return pd.DataFrame(out)


def pitcher_gap(mm, pid):
    t = mm["pitchers"].get("both")
    if t is None or pid not in t.index:
        return None
    r = t.loc[pid]
    return {"gap": float(r["gap_s"]), "raw": float(r["gap"]), "se": float(r["se"]), "w": float(r["w"]),
            "fb_share": float(r["fb_share"])}
