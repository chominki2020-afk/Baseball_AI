"""
제구 가정 백테스트: 추천 코스를 계산할 때 제구 오차를 어떻게 가정해야 실제로 더 맞나?

  A = 지금 앱: 모든 투수 0.5피트
  B = 리그 평균으로 고정 (build_command.py의 2024 추정, 약 0.69피트)
  C = 투수별 추정값 (2024, 리그 쪽으로 축소)
  - 모델·제구 추정 모두 2024만으로 → 2025 투구 8,000개 (backtest.py와 같은 표본·격자)
  - backtest.py의 제구 오차 격자(가운데 + 대각선 4점)를 가정값/0.5 배로 늘리거나 줄임
  - 결정적 비교: 두 추천 코스가 0.5피트 넘게 갈린 공에서, 실제 공이 어느 쪽 추천에 가까웠을 때 결과가 좋았나

실행: python backtest_command.py && python backtest_command.py --report
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import backtest as BT
import backtest_ensemble as BE
import build_matchup as B

CACHE = "backtest_reco_cmd.csv"


def recommend_batch(M, sit, scale):
    prof = M["profile"]
    keep = ["pitcher", "pitch_type", "group", "p_throws"] + [c for c in B.ARSENAL_F if c in M["arsenal"]]
    s = sit[["sid", "pitcher", "batter", "stand", "balls", "strikes", "outs", "r1", "r2", "r3",
             "sz_top", "sz_bot"]].merge(M["arsenal"][keep], on="pitcher").rename(columns={"pitch_type": "rec_type"})
    g = pd.DataFrame([(x, z, ex, ez) for x in BT.TX for z in BT.TZ for ex, ez in BT.NOISE], columns=["tx", "tz", "ex", "ez"])
    base = s.merge(g, how="cross")
    out = []
    for name, f in scale.items():                       # 가정마다 제구 오차 격자 크기만 바꿔서 같은 계산
        d = base.copy()
        k = f(d["pitcher"]) / 0.5
        zh = d["sz_top"] - d["sz_bot"]
        d["x_in"] = d["tx"] + d["ex"] * k
        d["z_norm"] = d["tz"] + d["ez"] * k / zh
        d["cell"] = B.cell_id(d["x_in"], d["z_norm"])
        d["xloc"] = np.digitize(d["x_in"], [-0.28, 0.28])
        d["same_hand"] = (d["stand"] == d["p_throws"]).astype(int)
        d["prev_grp"] = d["d_velo"] = d["d_z"] = np.nan
        d["group"] = d["group"].astype(str)
        d = B.attach(d, prof)
        d["pred"] = M["model"].predict(d[M["features"]])
        spot = d.groupby(["sid", "rec_type", "tx", "tz"])["pred"].mean().reset_index()
        b = spot.loc[spot.groupby(["sid", "rec_type"])["pred"].idxmin()]
        out.append(b.rename(columns={"pred": "best_rv", "tx": "rec_x", "tz": "rec_z"}).assign(v=name))
    return pd.concat(out, ignore_index=True)


def scales():
    C = pickle.load(open("command.pkl", "rb"))
    c24, m = C["c24"], C["c24_mean"]
    return {"A 0.5 고정": lambda p: pd.Series(0.5, index=p.index),
            f"B 리그 평균 {m:.2f}": lambda p: pd.Series(m, index=p.index),
            "C 투수별": lambda p: p.map(c24).fillna(m)}


def compute(M):
    df = pd.read_pickle("backtest_pitches.pkl")
    done = set(pd.read_csv(CACHE)["sid"]) if os.path.exists(CACHE) else set()
    todo = df[~df["sid"].isin(done)]
    sc, t0 = scales(), time.time()
    for i in range(0, len(todo), 100):
        part = recommend_batch(M, todo.iloc[i:i + 100], sc)
        part.to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False)
        print(f"  {min(i + 100, len(todo)):,} / {len(todo):,}  ({time.time() - t0:.0f}초)", flush=True)


def h2h(base, ra, rb, gap=0.5, seed=1):
    m = base.merge(ra, on=["sid", "pitch_type"]).merge(rb, on=["sid", "pitch_type"], suffixes=("_a", "_b"))
    h = m["sz_top"] - m["sz_bot"]
    g = np.hypot(m["rec_x_a"] - m["rec_x_b"], (m["rec_z_a"] - m["rec_z_b"]) * h)
    m, h = m[g > gap], h[g > gap]
    da = np.hypot(m["x_in"] - m["rec_x_a"], (m["z_norm"] - m["rec_z_a"]) * h)
    db = np.hypot(m["x_in"] - m["rec_x_b"], (m["z_norm"] - m["rec_z_b"]) * h)
    a, b = m[db < da]["y"], m[da < db]["y"]                 # a = 새 가정 쪽에 더 가까운 공
    rng = np.random.default_rng(seed)
    bs = [a.sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
          - b.sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean() for _ in range(1000)]
    return len(m), a.mean(), len(a), b.mean(), len(b), a.mean() - b.mean(), *np.percentile(bs, [2.5, 97.5])


def report():
    base = pd.read_pickle("backtest_pitches.pkl")
    base["y"] = BE.outcome(base)
    rec = pd.read_csv(CACHE)
    vs = list(dict.fromkeys(rec["v"]))
    R_ = {v: rec[rec["v"] == v].drop(columns="v").rename(columns={"rec_type": "pitch_type"}) for v in vs}
    rows = []
    print("[추천 코스와의 거리별 실점] 100구당 (투수·카운트 효과 제거), 낮을수록 투수에게 좋음")
    for v in vs:
        d = base.merge(R_[v], on=["sid", "pitch_type"])
        d["d"] = np.hypot(d["x_in"] - d["rec_x"], (d["z_norm"] - d["rec_z"]) * (d["sz_top"] - d["sz_bot"]))
        near, far = d[d["d"] < 0.5]["y"], d[d["d"] > 1.5]["y"]
        heart = ((R_[v]["rec_x"].abs() <= 0.28) & R_[v]["rec_z"].between(0.33, 0.67)).mean()
        print(f"  {v:<14} 0.5피트 이내 {near.mean():+6.2f} ({len(near):,}) | 1.5피트 넘게 {far.mean():+6.2f} ({len(far):,})"
              f" | 추천 코스가 한가운데 {heart:.0%}")
        rows.append({"version": v, "near": near.mean(), "n_near": len(near), "far": far.mean(), "heart": heart})
    print("\n[정면 비교] A(0.5 고정)와 추천이 0.5피트 넘게 갈린 공 — 새 가정 쪽에 더 가까웠을 때 − A 쪽에 더 가까웠을 때")
    for v in vs[1:]:
        n, am, na, bm, nb, diff, lo, hi = h2h(base, R_[vs[0]], R_[v])
        print(f"  {v:<14} 갈린 공 {n:,}개 | 새 가정 쪽 {am:+.2f} ({na:,}) vs A 쪽 {bm:+.2f} ({nb:,}) → {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        rows.append({"version": v, "vs": vs[0], "split_n": n, "h2h": diff, "lo": lo, "hi": hi})
    n, am, na, bm, nb, diff, lo, hi = h2h(base, R_[vs[1]], R_[vs[2]])
    print(f"  C vs B (투수별 vs 리그 평균) 갈린 공 {n:,}개 | 투수별 쪽 {am:+.2f} ({na:,}) vs 리그 평균 쪽 {bm:+.2f} ({nb:,}) "
          f"→ {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    rows.append({"version": vs[2], "vs": vs[1], "split_n": n, "h2h": diff, "lo": lo, "hi": hi})
    pd.DataFrame(rows).to_csv("val_command_backtest.csv", index=False)
    print("저장: val_command_backtest.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        compute(pickle.load(open("matchup_model.pkl", "rb")))
        print("추천 계산 완료 → python backtest_command.py --report")
