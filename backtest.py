"""
백테스트: 추천기 말대로 던지면 실제로 결과가 좋아지나?

  - 모델은 2024 데이터로만 학습 → 2025 시즌 실제 투구로 시험 (모델이 처음 보는 데이터)
  - 2025 투구마다, 그 상황(투수·타자·카운트·주자·아웃)에서 추천기가 고르는 구종별 최적 코스를 계산
  - 질문 1 (코스): 실제 공이 추천 코스에 가까울수록 실제 실점이 적었나?
           '그 투수가 평소 그 구종을 던지는 위치'에 가까운 것과 비교 → 제구 좋은 효과와 구분
  - 질문 2 (구종): 추천 1순위 구종을 던졌을 때가 다른 구종보다 결과가 좋았나?

실행 (계산이 무거워서 여러 번 나눠 돌려도 이어서 계산됨):
  python backtest.py            # 추천 계산 (끝날 때까지 반복 실행)
  python backtest.py --report   # 결과 분석
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import build_matchup as B

DATA = "statcast_2025-03-27_2025-09-28.csv"
CACHE = "backtest_reco.csv"
N_PITCH = 8000
TIME_LIMIT = 150          # 한 번 실행에 최대 몇 초 계산할지

# 백테스트용 성긴 격자 (추천기보다 거칠게: 계산량 절약)
TX = np.round(np.arange(-1.2, 1.21, 0.4), 2)            # 7칸
TZ = np.round(np.arange(-0.4, 1.36, 0.25), 3)           # 8칸
NOISE = [(0, 0), (0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)]   # 제구 오차(피트)


def sample_pitches(M):
    df = B.load(DATA)
    ars = M["arsenal"]
    known = set(zip(ars["pitcher"], ars["pitch_type"]))
    df = df[[(p, t) in known for p, t in zip(df["pitcher"], df["pitch_type"])]]
    df = df.sample(N_PITCH, random_state=7).reset_index(drop=True)
    df["sid"] = np.arange(len(df))
    return df


def recommend_batch(M, sit):
    """여러 상황을 한 번에: 상황 × 구종 × 목표지점 × 제구오차 → 모델 예측 → 구종별 최적 코스"""
    prof = M["profile"]
    keep = ["pitcher", "pitch_type", "group", "p_throws"] + [c for c in B.ARSENAL_F if c in M["arsenal"]]
    ars = M["arsenal"][keep]
    s = sit[["sid", "pitcher", "batter", "stand", "balls", "strikes", "outs", "r1", "r2", "r3",
             "sz_top", "sz_bot"]].merge(ars, on="pitcher")
    s = s.rename(columns={"pitch_type": "rec_type"})
    g = pd.DataFrame([(x, z, ex, ez) for x in TX for z in TZ for ex, ez in NOISE],
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
    d["pred"] = M["model"].predict(d[M["features"]])
    spot = d.groupby(["sid", "rec_type", "tx", "tz"])["pred"].mean().reset_index()
    best = spot.loc[spot.groupby(["sid", "rec_type"])["pred"].idxmin()]
    return best.rename(columns={"pred": "best_rv", "tx": "rec_x", "tz": "rec_z"})


def compute(M):
    if os.path.exists("backtest_pitches.pkl"):
        df = pd.read_pickle("backtest_pitches.pkl")
    else:
        df = sample_pitches(M)
        df.to_pickle("backtest_pitches.pkl")
    done = set(pd.read_csv(CACHE)["sid"]) if os.path.exists(CACHE) else set()
    todo = df[~df["sid"].isin(done)]
    print(f"추천 계산: 완료 {len(done):,} / {len(df):,}")
    t0 = time.time()
    for i in range(0, len(todo), 150):
        part = recommend_batch(M, todo.iloc[i:i + 150])
        part.to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False)
        done |= set(part["sid"])
        print(f"  {len(done):,} / {len(df):,}  ({time.time() - t0:.0f}초)")
        if time.time() - t0 > TIME_LIMIT:
            print("시간 제한 → 한 번 더 실행하면 이어서 계산해요")
            return False
    return True


def ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def report(M):
    df = pd.read_pickle("backtest_pitches.pkl")
    rec = pd.read_csv(CACHE)
    zh = df["sz_top"] - df["sz_bot"]

    # 투수가 평소 그 구종을 던지는 위치 (2024, 같은 카운트 유형·같은 타석 방향) = 비교 기준
    ref = B.load("statcast_2024-04-01_2024-09-30.csv")
    ref["ahead"] = np.sign(ref["strikes"] - ref["balls"])
    df["ahead"] = np.sign(df["strikes"] - df["balls"])
    typ = (ref.groupby(["pitcher", "pitch_type", "stand", "ahead"])[["x_in", "z_norm"]].mean()
              .rename(columns={"x_in": "typ_x", "z_norm": "typ_z"}).reset_index())
    df = df.merge(typ, on=["pitcher", "pitch_type", "stand", "ahead"], how="left")

    # 던진 구종의 추천 코스
    mine = rec.rename(columns={"rec_type": "pitch_type"})
    df = df.merge(mine, on=["sid", "pitch_type"], how="inner")
    df["d_rec"] = np.hypot(df["x_in"] - df["rec_x"], (df["z_norm"] - df["rec_z"]) * zh)
    df["d_typ"] = np.hypot(df["x_in"] - df["typ_x"], (df["z_norm"] - df["typ_z"]) * zh)
    df = df.dropna(subset=["d_typ"])

    # 투수 실력·카운트 차이를 빼고 비교 (같은 투수 안에서, 같은 카운트 안에서)
    y = df[B.TARGET] - df[B.TARGET].mean()
    for _ in range(6):
        for g in [["pitcher"], ["balls", "strikes"]]:
            y = y - y.groupby([df[k] for k in g]).transform("mean")
    df["y"] = y * 100
    print(f"\n2025 시즌 투구 {len(df):,}개로 백테스트 (모델은 2024만 학습)")
    print("숫자 = 100구당 실점 변화 (투수·카운트 효과 제거). 낮을수록 투수에게 좋음")

    def table(col, label):
        b = pd.cut(df[col], [0, 0.5, 1.0, 1.5, 99], labels=["0.5피트 이내", "0.5~1", "1~1.5", "1.5피트 넘게"])
        print(f"\n[{label}]")
        for k, part in df.groupby(b, observed=True):
            lo, hi = ci(part["y"])
            print(f"  {k:<12}{len(part):>6,}구   {part['y'].mean():+6.2f}   95% [{lo:+.2f}, {hi:+.2f}]")
    table("d_rec", "추천 코스와의 거리")
    table("d_typ", "평소 던지는 위치와의 거리 (비교용)")

    # 정면 비교: 추천 코스엔 가깝지만 평소 위치와는 먼 공 vs 그 반대
    a = df[(df["d_rec"] < 0.6) & (df["d_typ"] > 0.9)]["y"]
    b = df[(df["d_typ"] < 0.6) & (df["d_rec"] > 0.9)]["y"]
    diff = a.mean() - b.mean()
    rng = np.random.default_rng(1)
    bs = [a.sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
          - b.sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean() for _ in range(1000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("\n[정면 비교] 평소와 다르게 추천 코스로 간 공 vs 추천을 벗어나 평소대로 간 공")
    print(f"  추천 쪽 {len(a):,}구 {a.mean():+.2f}  /  평소 쪽 {len(b):,}구 {b.mean():+.2f}"
          f"  →  차이 {diff:+.2f}  95% [{lo:+.2f}, {hi:+.2f}]")

    # 구종 선택: 던진 구종이 추천 몇 순위였나
    rank = rec.copy()
    rank["rank"] = rank.groupby("sid")["best_rv"].rank(method="first")
    df = df.merge(rank[["sid", "rec_type", "rank"]].rename(columns={"rec_type": "pitch_type"}),
                  on=["sid", "pitch_type"])
    print("\n[구종 선택] 던진 구종이 추천 몇 순위였나")
    for k, part in df.groupby(np.where(df["rank"] == 1, "1순위", np.where(df["rank"] == 2, "2순위", "3순위 이하"))):
        lo, hi = ci(part["y"])
        print(f"  {k:<10}{len(part):>6,}구   {part['y'].mean():+6.2f}   95% [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    with open("matchup_model.pkl", "rb") as f:
        M = pickle.load(f)
    if a.report:
        report(M)
    elif compute(M):
        print("추천 계산 완료 → python backtest.py --report")
