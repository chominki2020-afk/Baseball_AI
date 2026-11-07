"""
가설 검증: "뜬공형 타자에게는 높은 직구가 낮은 직구보다 효과적이다"

설계
  - 타자 유형: 전반기(4~6월) 타구로 판정 → 후반기(7~9월) 투구로 검증 (끼워 맞추기 방지)
  - 비교 지표: (높은 직구 - 낮은 직구)의 run value 차이
               뜬공형에서 이 차이가 땅볼형보다 더 크게 '투수 유리'한가 = 이중차분(DiD)
  - 검증 1 (우연인가?): 타자 단위 부트스트랩 2,000번 → 95% 신뢰구간
  - 검증 2 (투수 실력 탓인가?): 같은 투수의 같은 구종, 같은 카운트끼리 비교되도록
               '투수×구종' 평균과 '카운트' 평균을 빼고(고정효과) 다시 계산

실행:
  python test_high_fastball.py
  python test_high_fastball.py --data statcast_2025-03-27_2025-09-28.csv --split 2025-07-01   # 다른 시즌 재현
"""
import argparse
import numpy as np
import pandas as pd

import build_matchup as B

N_BOOT = 2000
rng = np.random.default_rng(42)


def batter_type(first):
    """전반기 땅볼·뜬공 비율(축소 추정)로 타자를 3등분"""
    c = first[first["in_play"] == 1].dropna(subset=["launch_angle"]).copy()
    c["is_gb"] = (c["launch_angle"] < 10).astype(float)
    c["is_fb"] = c["launch_angle"].between(25, 50, inclusive="left").astype(float)
    k = 50
    agg = c.groupby("batter").agg(gb=("is_gb", "sum"), fb=("is_fb", "sum"), n=("is_gb", "size"))
    gb = (agg["gb"] + k * c["is_gb"].mean()) / (agg["n"] + k)
    fb = (agg["fb"] + k * c["is_fb"].mean()) / (agg["n"] + k)
    return pd.qcut(gb - fb, 3, labels=["뜬공형", "중간", "땅볼형"])


def demean(df, col, groups, iters=8):
    """여러 고정효과를 번갈아 빼기 (투수×구종, 카운트)"""
    r = df[col] - df[col].mean()
    for _ in range(iters):
        for g in groups:
            r = r - r.groupby([df[k] for k in g]).transform("mean")
    return r


def did_table(fb, col):
    """타자 × 높이 셀별 합계/개수 → 부트스트랩을 빠르게"""
    cell = fb.groupby(["batter", "type", "height"], observed=True)[col].agg(["sum", "count"])
    return cell.reset_index()


def estimate(cell, weights=None):
    c = cell if weights is None else cell.assign(sum=cell["sum"] * weights, count=cell["count"] * weights)
    t = c.groupby(["type", "height"], observed=True)[["sum", "count"]].sum()
    m = (t["sum"] / t["count"]).unstack("height") * 100
    gap = m["높은 공"] - m["낮은 공"]          # 음수일수록 높은 공이 투수에게 유리
    return gap["뜬공형"], gap["땅볼형"], gap["뜬공형"] - gap["땅볼형"]


def bootstrap(cell):
    batters = cell["batter"].unique()
    idx = {b: i for i, b in enumerate(batters)}
    pos = cell["batter"].map(idx).to_numpy()
    out = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(batters), len(batters)), minlength=len(batters))
        out.append(estimate(cell, w[pos]))
    return np.array(out)


def report(title, cell):
    est = estimate(cell)
    bs = bootstrap(cell)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"\n[{title}]  (100구당 실점 변화, 높은 공 - 낮은 공)")
    labels = ["뜬공형 타자", "땅볼형 타자", "차이 (뜬공형 - 땅볼형)"]
    for i, name in enumerate(labels):
        sig = "✔ 0을 포함 안 함" if hi[i] < 0 or lo[i] > 0 else "✘ 0을 포함 (우연일 수 있음)"
        print(f"  {name:<20}{est[i]:+6.2f}   95% 구간 [{lo[i]:+.2f}, {hi[i]:+.2f}]  {sig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="statcast_2024-04-01_2024-09-30.csv")
    ap.add_argument("--split", default="2024-07-01", help="이 날짜 전 = 타자 유형 판정, 이후 = 검증")
    a = ap.parse_args()

    df = B.load(a.data)
    first, second = df[df["game_date"] < a.split], df[df["game_date"] >= a.split]

    fb = second[second["grp"] == 0].copy()
    fb["type"] = fb["batter"].map(batter_type(first))
    fb["height"] = pd.cut(fb["z_norm"], [-9, 0.33, 0.67, 9], labels=["낮은 공", "가운데", "높은 공"])
    fb = fb.dropna(subset=["type"])
    fb = fb[fb["type"] != "중간"]
    fb["type"] = fb["type"].astype(str)
    print(f"후반기 패스트볼 {len(fb):,}개, 타자 {fb['batter'].nunique()}명 (뜬공형·땅볼형)")

    # 검증 1: 그대로
    report("검증 1: 단순 비교 + 부트스트랩", did_table(fb, "delta_run_exp"))

    # 검증 2: 같은 투수·구종, 같은 카운트 기준으로 보정
    fb["count"] = fb["balls"].astype(str) + "-" + fb["strikes"].astype(str)
    fb["rv_adj"] = demean(fb, "delta_run_exp", [["pitcher", "pitch_type"], ["count"]])
    report("검증 2: 투수×구종·카운트 효과 제거 후", did_table(fb, "rv_adj"))

    # 참고: 헛스윙률도 같은 방식으로 (스윙한 공만)
    sw = fb[fb["swing"] == 1].copy()
    sw["whiff_adj"] = demean(sw, "whiff", [["pitcher", "pitch_type"], ["count"]])
    e = estimate(did_table(sw, "whiff_adj"))
    print(f"\n[참고] 헛스윙률 차이 (높은 공 - 낮은 공, 보정 후, %p): "
          f"뜬공형 {e[0]:+.1f}, 땅볼형 {e[1]:+.1f}, 차이 {e[2]:+.1f}")
