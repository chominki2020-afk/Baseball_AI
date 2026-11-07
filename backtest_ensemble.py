"""
앙상블 백테스트: 실점 예측을 'LightGBM'만 쓸 때 vs '신경망+임베딩'과 평균할 때, 추천 코스가 실제로 더 맞나?

  - 둘 다 2024만으로 학습한 모델 (신경망 = embed_val2024.npz, 2025는 안 봄)
  - backtest.py와 같은 2025 투구 8,000개, 같은 격자·제구 오차로 구종별 최적 코스를 계산
  - w = 신경망 비중 (0 = 지금 앱, 0.5 = 평균, 1 = 신경망만)
  - 결정적 비교: 두 추천 코스가 0.5피트 넘게 갈린 공에서, 실제 공이 어느 쪽 추천에 더 가까웠을 때 결과가 좋았나

결과 (2025): 정면 비교에서 차이 없음 → 실점 순위는 LightGBM 유지, 스윙·헛스윙 확률만 평균 (AUC 검증 통과)

실행: python backtest_ensemble.py         (추천 계산, 2~3분. 이어서 계산됨)
      python backtest_ensemble.py --report
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import backtest as BT
import build_matchup as B
from build_embed import np_forward, prep

CACHE = "backtest_reco_ens.csv"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", 1e9))


def nn_index(ids, W, kind):
    mp = dict(zip(W[f"{kind}_ids"].tolist(), W[f"{kind}_idx"].tolist()))
    return pd.Series(ids).map(mp).fillna(0).astype(int).to_numpy()


def recommend_batch(M, W, sit):
    prof = M["profile"]
    keep = ["pitcher", "pitch_type", "group", "p_throws"] + [c for c in B.ARSENAL_F if c in M["arsenal"]]
    ars = M["arsenal"][keep]
    s = sit[["sid", "pitcher", "batter", "stand", "balls", "strikes", "outs", "r1", "r2", "r3",
             "sz_top", "sz_bot"]].merge(ars, on="pitcher")
    s = s.rename(columns={"pitch_type": "rec_type"})
    g = pd.DataFrame([(x, z, ex, ez) for x in BT.TX for z in BT.TZ for ex, ez in BT.NOISE],
                     columns=["tx", "tz", "ex", "ez"])
    d = s.merge(g, how="cross")
    zh = d["sz_top"] - d["sz_bot"]
    d["x_in"] = d["tx"] + d["ex"]
    d["z_norm"] = d["tz"] + d["ez"] / zh
    d["cell"] = B.cell_id(d["x_in"], d["z_norm"])
    d["xloc"] = np.digitize(d["x_in"], [-0.28, 0.28])
    d["same_hand"] = (d["stand"] == d["p_throws"]).astype(int)
    d["prev_grp"] = d["d_velo"] = d["d_z"] = np.nan
    d["group"] = d["group"].astype(str)
    d = B.attach(d, prof)
    d["lgb"] = M["model"].predict(d[M["features"]])
    X, _ = prep(d, {"mu": W["mu"], "sd": W["sd"]})
    _, _, d["nn"] = np_forward(W, X, nn_index(d["batter"], W, "b"), nn_index(d["pitcher"], W, "p"))
    spot = d.groupby(["sid", "rec_type", "tx", "tz"])[["lgb", "nn"]].mean().reset_index()
    out = []
    for w in WEIGHTS:
        spot["v"] = (1 - w) * spot["lgb"] + w * spot["nn"]
        b = spot.loc[spot.groupby(["sid", "rec_type"])["v"].idxmin(), ["sid", "rec_type", "tx", "tz", "v"]]
        out.append(b.rename(columns={"v": "best_rv", "tx": "rec_x", "tz": "rec_z"}).assign(w=w))
    return pd.concat(out, ignore_index=True)


def compute(M, W):
    df = pd.read_pickle("backtest_pitches.pkl")
    done = set(pd.read_csv(CACHE)["sid"]) if os.path.exists(CACHE) else set()
    todo = df[~df["sid"].isin(done)]
    print(f"추천 계산: {len(done):,} / {len(df):,}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), 100):
        part = recommend_batch(M, W, todo.iloc[i:i + 100])
        part.to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False)
        done |= set(part["sid"])
        print(f"  {len(done):,} / {len(df):,}  ({time.time() - t0:.0f}초)", flush=True)
        if time.time() - t0 > TIME_LIMIT:
            print("시간 제한 → 한 번 더 실행하면 이어서 계산해요")
            return False
    return True


def ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])


def outcome(df):
    """투수 실력·카운트 효과를 뺀 100구당 실점 (backtest.py와 같은 방식)"""
    y = df[B.TARGET] - df[B.TARGET].mean()
    for _ in range(6):
        for g in [["pitcher"], ["balls", "strikes"]]:
            y = y - y.groupby([df[k] for k in g]).transform("mean")
    return y * 100


def report():
    base = pd.read_pickle("backtest_pitches.pkl")
    base["y"] = outcome(base)
    zh = base["sz_top"] - base["sz_bot"]
    rec = pd.read_csv(CACHE)
    old = pd.read_csv("backtest_reco.csv")          # 지금 앱의 추천 (같은 계산이면 w=0과 같아야 함)
    chk = rec[rec["w"] == 0].merge(old, on=["sid", "rec_type"], suffixes=("", "_old"))
    same = ((chk["rec_x"] - chk["rec_x_old"]).abs() < 1e-6) & ((chk["rec_z"] - chk["rec_z_old"]).abs() < 1e-6)
    print(f"[확인] w=0 추천이 backtest_reco.csv와 같은 비율: {same.mean():.1%} ({len(chk):,}개) — 같은 매치업 모델이면 100%")

    rows, dist = [], {}
    print("\n[추천 코스와의 거리별 실점] 숫자 = 100구당 (투수·카운트 효과 제거), 낮을수록 투수에게 좋음")
    for w in WEIGHTS:
        r = rec[rec["w"] == w].rename(columns={"rec_type": "pitch_type"})
        d = base.merge(r, on=["sid", "pitch_type"], how="inner")
        d["d_rec"] = np.hypot(d["x_in"] - d["rec_x"], (d["z_norm"] - d["rec_z"]) * (d["sz_top"] - d["sz_bot"]))
        dist[w] = d.set_index("sid")["d_rec"]
        near, far = d[d["d_rec"] < 0.5]["y"], d[d["d_rec"] > 1.5]["y"]
        # 거리 1피트당 실점 기울기 (클수록 추천이 결과를 잘 가름)
        slope = np.polyfit(d["d_rec"].clip(0, 2.5), d["y"], 1)[0]
        bs = []
        rng = np.random.default_rng(0)
        for _ in range(500):
            s = d.sample(frac=1, replace=True, random_state=rng.integers(1e9))
            bs.append(np.polyfit(s["d_rec"].clip(0, 2.5), s["y"], 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  w={w:<4} 0.5피트 이내 {near.mean():+6.2f} ({len(near):,}구) | 1.5피트 넘게 {far.mean():+6.2f} ({len(far):,}구)"
              f" | 1피트 멀어질 때마다 {slope:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        rows.append({"w": w, "near": near.mean(), "n_near": len(near), "far": far.mean(), "n_far": len(far),
                     "slope": slope, "slope_lo": lo, "slope_hi": hi})

    # 결정적 비교: 추천이 갈린 공에서, 실제 공이 어느 쪽에 가까웠나
    print("\n[정면 비교] 두 추천 코스가 0.5피트 넘게 갈린 공 — 실제 공이 어느 추천에 더 가까웠을 때 결과가 좋았나")
    r0 = rec[rec["w"] == 0].rename(columns={"rec_type": "pitch_type"})
    for w in WEIGHTS[1:]:
        rw = rec[rec["w"] == w].rename(columns={"rec_type": "pitch_type"})
        m = base.merge(r0, on=["sid", "pitch_type"]).merge(rw, on=["sid", "pitch_type"], suffixes=("_0", "_w"))
        h = m["sz_top"] - m["sz_bot"]
        gap = np.hypot(m["rec_x_0"] - m["rec_x_w"], (m["rec_z_0"] - m["rec_z_w"]) * h)
        m = m[gap > 0.5]
        d0 = np.hypot(m["x_in"] - m["rec_x_0"], (m["z_norm"] - m["rec_z_0"]) * h[gap > 0.5])
        dw = np.hypot(m["x_in"] - m["rec_x_w"], (m["z_norm"] - m["rec_z_w"]) * h[gap > 0.5])
        a, b = m[dw < d0]["y"], m[d0 < dw]["y"]
        diff = a.mean() - b.mean()
        rng = np.random.default_rng(1)
        bs = [a.sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
              - b.sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean() for _ in range(1000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  w={w:<4} 갈린 공 {len(m):,}개 ({len(m) / len(base):.0%}) | 앙상블 쪽 {a.mean():+.2f} ({len(a):,}) vs "
              f"LightGBM 쪽 {b.mean():+.2f} ({len(b):,})  →  {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        rows.append({"w": w, "split_n": len(m), "ens_side": a.mean(), "lgb_side": b.mean(), "h2h": diff,
                     "h2h_lo": lo, "h2h_hi": hi})
    pd.DataFrame(rows).to_csv("val_ensemble_summary.csv", index=False)
    print("\n저장: val_ensemble_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        M = pickle.load(open("matchup_model.pkl", "rb"))
        z = np.load("embed_val2024.npz")
        W = {k: z[k] for k in z.files}
        if compute(M, W):
            print("추천 계산 완료 → python backtest_ensemble.py --report")
