"""
MLB 투수들은 미니맥스대로 던지나? — 무차별 원리 검증

미니맥스(제로섬 게임의 균형)라면, 투수가 '실제로 쓰는' 구종들은 같은 상황에서 결과(기대 실점)가 같아야 함.
  어떤 구종이 같은 상황에서 계속 더 좋다면 → 그 구종을 덜 던지고 있다는 뜻 (더 던지면 이득)
  계속 더 나쁘다면 → 너무 많이 던지고 있다는 뜻

방법
  같은 상황 = 같은 투수 × 같은 카운트 × 같은 손 대결(좌우)  ← 이 안에서만 비교 (고정효과)
  결과 = 실제 실점 가치 (rv_ctx, 카운트 변화까지 포함한 득점 기대 변화)
  패스트볼 / 변화구 / 오프스피드 계열 간 차이 → 투수 단위로 재추출한 95% 구간
  2024와 2025에서 각각 → 재현되나?
  투수별 차이는 리그 쪽으로 축소 보정 → 2024 투수별 차이가 2025에도 이어지나? (진짜 성향인가, 운인가)

실행: python build_minimax.py   (3~5분)
"""
import pickle

import numpy as np
import pandas as pd

import build_matchup as B

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
GRP = {0: "패스트볼", 1: "변화구", 2: "오프스피드"}
MIN_P = 500


def prep(year):
    d = B.load(FILES[year])
    d = d[d.groupby("pitcher")["pitch_type"].transform("size") >= MIN_P].copy()
    d["count"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    d["cell"] = d["pitcher"].astype(str) + "_" + d["count"] + "_" + d["same_hand"].astype(int).astype(str)
    d["rv"] = d[B.TARGET] * 100                                    # 100구당
    for g in (1, 2):
        d[f"g{g}"] = (d["grp"] == g).astype(float)
    return d


def fe_gaps(d, B_=300, seed=0):
    """같은 칸(투수×카운트×좌우) 안에서 변화구·오프스피드가 패스트볼보다 몇 점 나은지 (음수 = 패스트볼보다 실점 적음)"""
    cols = ["g1", "g2"]
    X = d[cols] - d.groupby("cell")[cols].transform("mean")
    y = d["rv"] - d.groupby("cell")["rv"].transform("mean")
    XtX = X.T.to_numpy() @ X.to_numpy()
    b = np.linalg.solve(XtX, X.T.to_numpy() @ y.to_numpy())
    # 투수 단위 부트스트랩
    P = pd.DataFrame(np.c_[X.to_numpy(), y.to_numpy()], columns=["x1", "x2", "y"])
    P["p"] = d["pitcher"].values
    G = pd.DataFrame({"x1x1": P["x1"] ** 2, "x1x2": P["x1"] * P["x2"], "x2x2": P["x2"] ** 2,
                      "x1y": P["x1"] * P["y"], "x2y": P["x2"] * P["y"], "p": P["p"]}).groupby("p").sum()
    rng = np.random.default_rng(seed)
    ps = G.index.to_numpy()
    bs = []
    for _ in range(B_):
        s = G.loc[rng.choice(ps, len(ps))].sum()
        A = np.array([[s["x1x1"], s["x1x2"]], [s["x1x2"], s["x2x2"]]])
        bs.append(np.linalg.solve(A, [s["x1y"], s["x2y"]]))
    bs = np.array(bs)
    return b, np.percentile(bs, [2.5, 97.5], axis=0)


def pitcher_gaps(d):
    """투수별: 같은 칸 안에서 '패스트볼 − 나머지' 차이와 표준오차 → 리그 쪽으로 축소"""
    d = d.copy()
    d["nfb"] = (d["grp"] != 0).astype(float)
    x = d["nfb"] - d.groupby("cell")["nfb"].transform("mean")
    y = d["rv"] - d.groupby("cell")["rv"].transform("mean")
    t = pd.DataFrame({"p": d["pitcher"], "xy": x * y, "xx": x * x, "yy": y * y}).groupby("p").sum()
    n = d.groupby("pitcher").size()
    t["gap"] = -t["xy"] / t["xx"]                                   # 패스트볼 − 나머지 (+면 패스트볼이 더 나쁨)
    resid_var = (t["yy"] - t["xy"] ** 2 / t["xx"]) / n
    t["se"] = np.sqrt(resid_var / t["xx"])
    t = t[np.isfinite(t["gap"]) & (t["xx"] > 5)]
    m = np.average(t["gap"], weights=1 / t["se"] ** 2)
    tau2 = max(np.var(t["gap"]) - np.mean(t["se"] ** 2), 1e-6)
    t["w"] = tau2 / (tau2 + t["se"] ** 2)
    t["gap_s"] = t["w"] * t["gap"] + (1 - t["w"]) * m
    t["fb_share"] = d.groupby("pitcher")["grp"].apply(lambda s: (s == 0).mean())
    t["name"] = d.groupby("pitcher")["player_name"].first()
    return t[["name", "gap", "se", "gap_s", "w", "fb_share"]], m, tau2


if __name__ == "__main__":
    out = {"league": {}, "by_count": {}, "pitchers": {}}
    data = {}
    for yr in (2024, 2025):
        d = prep(yr)
        data[yr] = d
        b, ci = fe_gaps(d)
        out["league"][yr] = {"g1": (b[0], ci[:, 0].tolist()), "g2": (b[1], ci[:, 1].tolist()),
                             "n": len(d), "pitchers": int(d["pitcher"].nunique()),
                             "share": d["grp"].value_counts(normalize=True).sort_index().tolist()}
        print(f"\n=== {yr}  투수 {d['pitcher'].nunique()}명, 투구 {len(d):,}개  (구종 비율 FB/BR/OFF = "
              f"{', '.join(f'{x:.0%}' for x in out['league'][yr]['share'])})")
        print(f"  같은 투수·카운트·좌우 안에서 패스트볼 대비 100구당 실점 차이 (음수 = 패스트볼보다 좋음)")
        print(f"    변화구    {b[0]:+.2f}  [{ci[0, 0]:+.2f}, {ci[1, 0]:+.2f}]")
        print(f"    오프스피드 {b[1]:+.2f}  [{ci[0, 1]:+.2f}, {ci[1, 1]:+.2f}]")
        rows = []
        for c, g in d.groupby("count"):
            if g["pitcher"].nunique() < 100:
                continue
            bb, cc = fe_gaps(g, B_=150)
            rows.append({"count": c, "n": len(g), "fb_share": (g["grp"] == 0).mean(), "g1": bb[0], "g1_lo": cc[0, 0],
                         "g1_hi": cc[1, 0], "g2": bb[1], "g2_lo": cc[0, 1], "g2_hi": cc[1, 1]})
        bc = pd.DataFrame(rows).sort_values("count")
        out["by_count"][yr] = bc
        print(bc.round(2).to_string(index=False))
        pg, m, tau2 = pitcher_gaps(d)
        out["pitchers"][yr] = pg
        print(f"  투수별 '패스트볼 − 나머지' 평균 {m:+.2f}, 진짜 투수 간 차이 표준편차 {np.sqrt(tau2):.2f} "
              f"(평균 개인 데이터 반영 비율 {pg['w'].mean():.0%})")
    a, b_ = out["pitchers"][2024], out["pitchers"][2025]
    j = a.join(b_, lsuffix="_24", rsuffix="_25", how="inner")
    r_raw = np.corrcoef(j["gap_24"], j["gap_25"])[0, 1]
    r_s = np.corrcoef(j["gap_s_24"], j["gap_25"])[0, 1]
    # 2024에 '패스트볼이 더 나빴던' 투수는 2025에 패스트볼을 줄였나?
    fb_change = np.corrcoef(j["gap_s_24"], j["fb_share_25"] - j["fb_share_24"])[0, 1]
    print(f"\n[투수별 차이가 이어지나] 두 시즌 다 있는 투수 {len(j)}명: 2024→2025 상관 {r_raw:+.3f} (축소 보정 후 {r_s:+.3f})")
    print(f"  2024에 패스트볼이 상대적으로 나빴던 투수일수록 2025에 패스트볼 비율을 줄였나? 상관 {fb_change:+.3f}")
    out["persist"] = {"n": len(j), "r_raw": r_raw, "r_shrunk": r_s, "fb_change": fb_change}
    # 앱용: 두 시즌 합쳐서 투수별 (축소 보정)
    allp, m, tau2 = pitcher_gaps(pd.concat([data[2024], data[2025]], ignore_index=True).assign(
        cell=lambda x: x["cell"] + "_" + x["game_date"].astype(str).str[:4]))
    out["pitchers"]["both"] = allp
    out["pitchers"]["both_mean"] = m
    with open("minimax.pkl", "wb") as f:
        pickle.dump(out, f)
    print("저장: minimax.pkl")
