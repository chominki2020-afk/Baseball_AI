"""
투수별 제구 오차 추정 — 지금까지는 모든 투수를 '목표에서 평균 0.5피트 빗나감'으로 가정했다 (사례 6)

문제: Statcast에는 '노린 곳'(포수 미트 위치)이 없다. 그래서 흩어진 정도 = 의도적으로 바꾼 코스 + 실제 제구 오차.
방법: 의도가 거의 하나인 공만 쓴다 → 3-0, 3-1, 2-0 카운트의 패스트볼 (볼넷 피하려고 '스트라이크 넣기')
      같은 투수 · 같은 타석 방향 안에서 평균 위치로부터 흩어진 정도(좌우·상하 표준편차 평균, 피트) = 제구 오차
      표본이 적은 투수는 리그 평균 쪽으로 축소 (경험적 베이즈)
검증: ① 시즌 안 반분 신뢰도 (홀수/짝수 경기)  ② 2024 → 2025 상관 (진짜 성향인가)
      ③ 2025 볼넷률을 예측하나 — 2024 볼넷률만 쓸 때보다 정보가 더 있나

실행: python build_command.py   (1~2분) → command.pkl, val_command_summary.csv
"""
import pickle

import numpy as np
import pandas as pd

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
FB = ("FF", "SI", "FC")
COUNTS = ((3, 0), (3, 1), (2, 0))
MIN_N = 15


def load(year):
    d = pd.read_csv(FILES[year], usecols=["pitcher", "player_name", "game_pk", "at_bat_number", "balls", "strikes",
                                         "pitch_type", "plate_x", "plate_z", "stand", "description", "events"])
    d["season"] = year
    return d


def spread(d):
    """투수별: 같은 타석 방향 안 평균 위치에서의 흩어짐 (좌우·상하 표준편차의 평균, 피트)"""
    k = list(zip(d["balls"], d["strikes"]))
    m = d["pitch_type"].isin(FB) & pd.Series([c in COUNTS for c in k], index=d.index) & \
        ~d["description"].isin(["pitchout", "intent_ball"]) & d["plate_x"].notna()
    s = d[m].copy()
    g = s.groupby(["pitcher", "stand"])
    s["rx"] = s["plate_x"] - g["plate_x"].transform("mean")
    s["rz"] = s["plate_z"] - g["plate_z"].transform("mean")
    s["dof"] = 1 - 1 / g["plate_x"].transform("size")              # 평균을 뺀 만큼 자유도 보정
    t = s.groupby("pitcher").agg(n=("rx", "size"), sxx=("rx", lambda v: (v ** 2).sum()),
                                 szz=("rz", lambda v: (v ** 2).sum()), dof=("dof", "sum"))
    t["sd_x"] = np.sqrt(t["sxx"] / t["dof"].clip(1))
    t["sd_z"] = np.sqrt(t["szz"] / t["dof"].clip(1))
    t["c"] = (t["sd_x"] + t["sd_z"]) / 2
    t["se"] = t["c"] / np.sqrt(4 * t["dof"].clip(1))                # 두 축 평균한 표준편차의 표본 오차
    return t[t["n"] >= MIN_N][["n", "sd_x", "sd_z", "c", "se"]]


def shrink(t):
    w_ = 1 / t["se"] ** 2
    m = np.average(t["c"], weights=w_)
    tau2 = max(np.var(t["c"]) - np.mean(t["se"] ** 2), 1e-6)
    t = t.copy()
    t["w"] = tau2 / (tau2 + t["se"] ** 2)
    t["c_s"] = t["w"] * t["c"] + (1 - t["w"]) * m
    return t, m, np.sqrt(tau2)


def walk_rate(d):
    pa = d.dropna(subset=["events"])
    pa = pa[pa["events"] != "intent_walk"]
    g = pa.groupby("pitcher")
    return pd.DataFrame({"pa": g.size(), "bb": g["events"].apply(lambda e: (e == "walk").mean())})


def corr(a, b):
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def boot_r(a, b, B=1000, seed=0):
    ok = a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    rng = np.random.default_rng(seed)
    rs = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
    return np.percentile(rs, [2.5, 97.5])


if __name__ == "__main__":
    data = {y: load(y) for y in FILES}
    out, rows = {}, []
    for y, d in data.items():
        t, m, tau = shrink(spread(d))
        out[y] = t
        print(f"[{y}] 투수 {len(t)}명 (3볼·2-0 패스트볼 {MIN_N}개 이상, 중앙값 {t['n'].median():.0f}개) | "
              f"제구 오차 리그 평균 {m:.2f}피트, 투수 간 진짜 차이 표준편차 {tau:.2f} (개인 데이터 반영 {t['w'].mean():.0%})")
        rows.append({"check": f"{y} 리그 평균 (피트)", "value": m, "lo": np.nan, "hi": np.nan, "n": len(t)})
        rows.append({"check": f"{y} 투수 간 차이 SD", "value": tau, "lo": np.nan, "hi": np.nan, "n": len(t)})
        # ① 반분 신뢰도
        odd = d[d["game_pk"] % 2 == 1]
        even = d[d["game_pk"] % 2 == 0]
        a, b = spread(odd)["c"], spread(even)["c"]
        r, n = corr(a, b.reindex(a.index))
        sb = 2 * r / (1 + r)
        print(f"  ① 반분 신뢰도 (홀수/짝수 경기): r = {r:.2f} → 한 시즌 전체 {sb:.2f} (투수 {n}명)")
        rows.append({"check": f"{y} 반분 신뢰도 (시즌 전체 환산)", "value": sb, "lo": np.nan, "hi": np.nan, "n": n})

    # ② 2024 → 2025
    j = out[2024].join(out[2025], lsuffix="_24", rsuffix="_25", how="inner")
    r, n = corr(j["c_24"], j["c_25"])
    lo, hi = boot_r(j["c_24"], j["c_25"])
    print(f"② 2024 → 2025 제구 오차 상관: r = {r:.2f} [{lo:.2f}, {hi:.2f}] (투수 {n}명)")
    rows.append({"check": "2024→2025 상관", "value": r, "lo": lo, "hi": hi, "n": n})

    # ③ 2025 볼넷률 예측
    w24, w25 = walk_rate(data[2024]), walk_rate(data[2025])
    v = out[2024][["c_s"]].join(w24.add_suffix("_24"), how="inner").join(w25.add_suffix("_25"), how="inner")
    v = v[(v["pa_24"] >= 150) & (v["pa_25"] >= 150)]
    r1, n1 = corr(v["c_s"], v["bb_25"])
    lo1, hi1 = boot_r(v["c_s"], v["bb_25"])
    r0, _ = corr(v["bb_24"], v["bb_25"])
    print(f"③ 2024 제구 오차 → 2025 볼넷률: r = {r1:+.2f} [{lo1:+.2f}, {hi1:+.2f}] (투수 {n1}명, 양쪽 150타석 이상)")
    print(f"   비교: 2024 볼넷률 → 2025 볼넷률 r = {r0:+.2f}")
    # 2024 볼넷률을 이미 알 때 제구 오차가 더해 주는 정보 (표준화 회귀 계수, 투수 단위 부트스트랩)
    Z = (v - v.mean()) / v.std()
    X = np.c_[np.ones(len(Z)), Z["bb_24"], Z["c_s"]]
    beta = np.linalg.lstsq(X, Z["bb_25"], rcond=None)[0]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(Z), len(Z))
        bs.append(np.linalg.lstsq(X[i], Z["bb_25"].to_numpy()[i], rcond=None)[0][2])
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    r2_0 = r0 ** 2
    pred = X @ beta
    r2_1 = 1 - ((Z["bb_25"] - pred) ** 2).sum() / ((Z["bb_25"] - Z["bb_25"].mean()) ** 2).sum()
    print(f"   2024 볼넷률을 알 때 제구 오차의 추가 효과: 표준화 계수 {beta[2]:+.2f} [{blo:+.2f}, {bhi:+.2f}], "
          f"설명력 R² {r2_0:.3f} → {r2_1:.3f}")
    rows += [{"check": "2024 제구 → 2025 볼넷률", "value": r1, "lo": lo1, "hi": hi1, "n": n1},
             {"check": "2024 볼넷률 → 2025 볼넷률", "value": r0, "lo": np.nan, "hi": np.nan, "n": n1},
             {"check": "볼넷률 통제 후 제구 계수", "value": beta[2], "lo": blo, "hi": bhi, "n": n1},
             {"check": "R² (볼넷률만 → +제구)", "value": r2_1, "lo": r2_0, "hi": np.nan, "n": n1}]
    pd.DataFrame(rows).to_csv("val_command_summary.csv", index=False)

    # 앱용: 2024 값 (모델과 같은 시즌, 백테스트용) + 두 시즌 합친 값
    both = pd.concat([data[2024], data[2025].assign(game_pk=data[2025]["game_pk"])], ignore_index=True)
    tb, mb, _ = shrink(spread(both))
    names = both.groupby("pitcher")["player_name"].first()
    res = {"c24": out[2024]["c_s"].to_dict(), "c24_mean": float(np.average(out[2024]["c"], weights=1 / out[2024]["se"] ** 2)),
           "c_all": tb["c_s"].to_dict(), "c_all_mean": float(mb),
           "table": tb.join(names.rename("name"))}
    pickle.dump(res, open("command.pkl", "wb"))
    best = tb.join(names.rename("name")).sort_values("c_s")
    print("\n제구 오차 작은 투수 (두 시즌):", ", ".join(f"{n} {c:.2f}" for n, c in zip(best["name"][:5], best["c_s"][:5])))
    print("제구 오차 큰 투수:", ", ".join(f"{n} {c:.2f}" for n, c in zip(best["name"][-5:], best["c_s"][-5:])))
    print("저장: command.pkl, val_command_summary.csv")
