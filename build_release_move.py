"""
릴리스를 바꾸면 공(무브먼트·구속·회전)은 어떻게 변하나 — 실제 데이터에서 배우기

두 가지 방법으로 같은 질문을 따로 풀어서 서로 맞는지 확인
  A) 시즌 간 변화: 2024 → 2025 같은 투수·같은 구종의 평균 릴리스 변화 vs 평균 무브먼트 변화
     (팔 각도를 일부러 바꾼 투수들이 여기서 신호를 만듦)
  B) 경기 간 변화: 한 시즌 안에서 경기마다 평균 릴리스가 조금씩 다를 때 무브먼트가 같이 움직이나
     (투수·구종 고정효과 + 구장 고정효과로 구장마다 다른 측정 오차 제거)

결과: release_move.pkl  = 구종별 계수 (릴리스 1피트 바뀔 때 무브먼트 몇 인치 / 구속 몇 mph / 회전 몇 rpm)
실행: python build_release_move.py   (3~5분)
"""
import pickle

import numpy as np
import pandas as pd

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
COLS = ["pitcher", "player_name", "p_throws", "pitch_type", "game_pk", "home_team", "release_pos_x",
        "release_pos_z", "release_extension", "pfx_x", "pfx_z", "release_speed", "release_spin_rate"]
X = ["rel_z", "rel_side", "ext"]
Y = ["ivb", "hb", "velo", "spin"]
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS"]
GROUP = {"FF": "fb", "SI": "fb", "FC": "fb", "SL": "brk", "ST": "brk", "CU": "brk", "KC": "brk", "SV": "brk",
         "CS": "brk", "CH": "off", "FS": "off", "FO": "off", "SC": "off"}


def read(year):
    d = pd.read_csv(FILES[year], usecols=COLS).dropna()
    d = d[d["pitch_type"].isin(GROUP)]
    d["season"] = year
    d["rel_z"] = d["release_pos_z"]
    d["rel_side"] = d["release_pos_x"].abs()
    d["ext"] = d["release_extension"]
    d["hb"] = np.where(d["p_throws"] == "L", -d["pfx_x"], d["pfx_x"]) * 12   # 모델과 같은 방향 (-면 팔 쪽)
    d["ivb"] = d["pfx_z"] * 12
    d["velo"] = d["release_speed"]
    d["spin"] = d["release_spin_rate"]
    return d


def wls(Xm, y, w):
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(Xm * W, y * W[:, 0], rcond=None)
    res = y - Xm @ coef
    # 투수 단위로 묶지 않은 단순 표준오차 (참고용)
    s2 = (w * res ** 2).sum() / max(len(y) - Xm.shape[1], 1)
    cov = s2 * np.linalg.pinv((Xm * W).T @ (Xm * W))
    return coef, np.sqrt(np.diag(cov))


def season_change(df):
    """A) 2024 → 2025 같은 투수·구종 평균 변화"""
    g = (df.groupby(["pitcher", "pitch_type", "season"])
           .agg(n=("velo", "size"), **{c: (c, "mean") for c in X + Y}).reset_index())
    g = g[g["n"] >= 100]
    a = g[g["season"] == 2024].set_index(["pitcher", "pitch_type"])
    b = g[g["season"] == 2025].set_index(["pitcher", "pitch_type"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    out = pd.DataFrame({c: j[c + "_b"] - j[c + "_a"] for c in X + Y})
    out["w"] = 1 / (1 / j["n_a"] + 1 / j["n_b"])
    return out.reset_index()


def game_level(df):
    """B) 경기별 평균에서 투수·구종 고정효과와 구장 고정효과를 번갈아 빼기"""
    g = (df.groupby(["season", "pitcher", "pitch_type", "game_pk", "home_team"])
           .agg(n=("velo", "size"), **{c: (c, "mean") for c in X + Y}).reset_index())
    g = g[g["n"] >= 8]
    key = g["season"].astype(str) + "_" + g["pitcher"].astype(str) + "_" + g["pitch_type"]
    park = g["season"].astype(str) + "_" + g["home_team"]
    v = g[X + Y].copy()
    for _ in range(8):
        v = v - v.groupby(key).transform("mean")
        v = v - v.groupby(park).transform("mean")
    v["pitch_type"] = g["pitch_type"].values
    v["w"] = g["n"].values
    return v


def fit(d, label):
    rows = []
    for t in TYPES + ["fb", "brk", "off"]:
        s = d[d["pitch_type"] == t] if t in TYPES else d[d["pitch_type"].map(GROUP) == t]
        if len(s) < 60:
            continue
        # 큰 이상치(구종 분류 바뀜 등) 제외
        s = s[(s[X].abs() < 1.0).all(axis=1) & (s["ivb"].abs() < 8) & (s["hb"].abs() < 8)]
        Xm = s[X].to_numpy()
        for y in Y:
            c, se = wls(Xm, s[y].to_numpy(), s["w"].to_numpy())
            rows.append({"method": label, "type": t, "y": y, "n": len(s),
                         **{f"b_{x}": c[i] for i, x in enumerate(X)},
                         **{f"se_{x}": se[i] for i, x in enumerate(X)}})
    return pd.DataFrame(rows)


def cached(year):
    import os
    p = f"rm_{year}.pkl"
    if not os.path.exists(p):
        d = read(year)
        d[["season", "pitcher", "pitch_type", "game_pk", "home_team"] + X + Y].to_pickle(p)
    return pd.read_pickle(p)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "read":          # 오래 걸리면 연도별로 먼저 읽어 두기
        cached(int(sys.argv[2]))
        print("읽기 완료", sys.argv[2])
        sys.exit()
    df = pd.concat([cached(2024), cached(2025)], ignore_index=True)
    print(f"읽음: {len(df):,}구")
    A = fit(season_change(df), "season")
    B = fit(game_level(df), "game")
    res = pd.concat([A, B], ignore_index=True)
    pd.set_option("display.width", 220)
    for y in Y:
        print(f"\n=== {y}  (릴리스 1피트 변화당)")
        t = res[res["y"] == y].pivot(index="type", columns="method", values=["b_rel_z", "b_rel_side", "b_ext"])
        print(t.round(2).to_string())
    print("\n시즌 간 표본 수:", A.drop_duplicates("type").set_index("type")["n"].to_dict())
    with open("release_move.pkl", "wb") as f:
        pickle.dump(res, f)
    print("\n저장: release_move.pkl")
