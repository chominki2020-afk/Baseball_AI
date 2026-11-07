"""
릴리스 변화 추천 검증 — 2024 → 2025 실제로 릴리스를 바꾼 투수들 (자연 실험)

질문: 2024 데이터만 보고 "릴리스를 이만큼 바꾸면 헛스윙이 이만큼 변한다"고 예측한 게
      2025에 실제로 그만큼 바꾼 투수들에게서 맞았나?

예측 세 가지 (모두 2024 정보만 사용, 실제 릴리스 변화량만 2025에서 가져옴)
  P0 릴리스만     : 무브먼트·구속 그대로
  P1 + 학습 무브  : 2024 경기별 변화에서만 배운 릴리스→무브 관계로 무브먼트도 바꿈 (2025 정보 누출 없음)
  P2 + 실제 무브  : 2025에 실제로 바뀐 무브먼트·구속·회전을 넣음 (스윙 모델 자체가 맞는지만 따로 보기)
실제값: 2025 SwStr% − 2024 SwStr% (그 구종 리그 전체 변화는 빼고)

실행: python validate_release.py prep   (데이터 요약, 1~2분)
      python validate_release.py eval   (예측·비교)
"""
import pickle
import sys

import numpy as np
import pandas as pd

import build_release_move as RM
import mechanics as MC

COLS = RM.COLS + ["description", "home_team", "away_team", "inning_topbot"]
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CHANGE = 0.25          # 이 이상(피트, 약 8cm) 바뀌면 '바꾼 투수'


def season_summary(year):
    d = pd.read_csv(RM.FILES[year], usecols=COLS)
    d = d.dropna(subset=RM.COLS)
    d = d[d["pitch_type"].isin(RM.GROUP)]
    d["rel_z"] = d["release_pos_z"]
    d["rel_side"] = d["release_pos_x"].abs()
    d["ext"] = d["release_extension"]
    d["hb"] = np.where(d["p_throws"] == "L", -d["pfx_x"], d["pfx_x"]) * 12
    d["ivb"] = d["pfx_z"] * 12
    d["velo"] = d["release_speed"]
    d["spin"] = d["release_spin_rate"]
    d["whiff"] = d["description"].isin(WHIFF)
    d["team"] = np.where(d["inning_topbot"] == "Top", d["home_team"], d["away_team"])
    team = d.groupby("pitcher")["team"].agg(lambda s: s.value_counts().index[0])
    g = (d.groupby(["pitcher", "pitch_type"])
           .agg(n=("velo", "size"), whiffs=("whiff", "sum"),
                **{c: (c, "mean") for c in RM.X + RM.Y}).reset_index())
    g["swstr"] = g["whiffs"] / g["n"]
    g["team"] = g["pitcher"].map(team)
    g["name"] = g["pitcher"].map(d.groupby("pitcher")["player_name"].first())
    games = RM.game_level(d.assign(season=year)) if year == 2024 else None
    return g, games


def corr_ci(x, y, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r, np.percentile(bs, [2.5, 97.5])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if step == "prep":
        a, games = season_summary(2024)
        b, _ = season_summary(2025)
        coef = RM.fit(games, "game")                     # 2024 안에서만 배운 릴리스→무브 관계
        with open("val_release_prep.pkl", "wb") as f:
            pickle.dump({"a": a, "b": b, "coef": coef}, f)
        print("prep 완료", len(a), len(b))
        sys.exit()

    P = pickle.load(open("val_release_prep.pkl", "rb"))
    S = pickle.load(open("swing_model.pkl", "rb"))
    samp = pickle.load(open("pitch_samples.pkl", "rb"))
    phys = pickle.load(open("physics.pkl", "rb")).set_index(["pitcher", "pitch_type"])
    move24 = {(r["type"], r["y"]): np.array([r["b_rel_z"], r["b_rel_side"], r["b_ext"]])
              for _, r in P["coef"].iterrows() if r["y"] in ("ivb", "hb")}

    a = P["a"][P["a"]["n"] >= 100].set_index(["pitcher", "pitch_type"])
    b = P["b"][P["b"]["n"] >= 100].set_index(["pitcher", "pitch_type"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    for c in RM.X + RM.Y + ["swstr"]:
        j["d_" + c] = j[c + "_b"] - j[c + "_a"]
    # 리그 전체 해마다 변화(측정 보정, 리그 경향)는 구종별 중앙값을 빼서 제거
    for c in ["d_swstr"] + ["d_" + x for x in RM.X]:
        j[c] = j[c] - j.groupby(level="pitch_type")[c].transform("median")
    j["moved"] = j[["d_rel_z", "d_rel_side", "d_ext"]].abs().max(axis=1) >= CHANGE
    j["traded"] = j["team_a"] != j["team_b"]
    keys = set(map(tuple, samp[["pitcher", "pitch_type"]].drop_duplicates().to_numpy()))
    j = j[[k in keys and k in phys.index for k in j.index]]
    print(f"2024·2025 둘 다 100구 이상 + 2024 샘플 있음: {len(j)}개 (투수×구종),  바꾼 경우: {j['moved'].sum()}개")

    # ---- 예측: 해당 투수·구종의 2024 실제 투구들에 변화량만 넣어 SwStr 비교
    sel = j.reset_index()
    base = samp.merge(sel[["pitcher", "pitch_type"]], on=["pitcher", "pitch_type"])
    X0 = base[S["features"]]
    p0 = S["swing"].predict_proba(X0)[:, 1] * S["whiff"].predict_proba(X0)[:, 1]
    out = {"base": pd.Series(p0).groupby([base["pitcher"].values, base["pitch_type"].values]).mean()}
    PRED = ("P0", "P1", "P2", "P3", "P4", "P5")
    for name in PRED:
        dd = {}
        for _, r in sel.iterrows():
            k = (r["pitcher"], r["pitch_type"])
            dh, ds, de = r["d_rel_z"], r["d_rel_side"], r["d_ext"]
            if name == "P0":
                mv = {"ivb": 0.0, "hb": 0.0, "velo": 0.0, "spin": 0.0}
            elif name in ("P1", "P4", "P5"):
                m = MC.move_delta(move24, r["pitch_type"], dh, ds, de)
                mv = {"ivb": m["ivb"], "hb": m["hb"], "velo": 0.0, "spin": 0.0}
            else:
                mv = {"ivb": r["d_ivb"], "hb": r["d_hb"], "velo": r["d_velo"], "spin": r["d_spin"]}
            if name in ("P3", "P4"):                         # 공 모양 변화만 (릴리스는 그대로)
                dh = ds = de = 0.0
            mv["vaa"] = MC.vaa_shift(phys.loc[k].to_dict(), dh, ds, de, d_ivb=mv["ivb"])
            if name == "P5":                                 # 진입각은 릴리스 기하학까지 반영, 릴리스 값 자체는 모델에 안 넣음
                dh = ds = de = 0.0
            dd[k] = (dh, ds, de, mv)
        x = base.copy()
        dk = [dd[(p, t)] for p, t in zip(x["pitcher"], x["pitch_type"])]
        x["rel_z"] += [d[0] for d in dk]
        x["rel_side"] += [d[1] for d in dk]
        x["ext"] += [d[2] for d in dk]
        x["vaa_rel"] += [d[3]["vaa"] for d in dk]
        x["pfx_z"] += [d[3]["ivb"] for d in dk]
        x["pfx_x"] += [d[3]["hb"] for d in dk]
        x["release_speed"] += [d[3]["velo"] for d in dk]
        x["release_spin_rate"] += [d[3]["spin"] for d in dk]
        X = x[S["features"]]
        p = S["swing"].predict_proba(X)[:, 1] * S["whiff"].predict_proba(X)[:, 1]
        out[name] = pd.Series(p).groupby([x["pitcher"].values, x["pitch_type"].values]).mean()
    for name in PRED:
        j["pred_" + name] = ((out[name] - out["base"]) * 100).reindex(j.index).values
        j["pred_" + name] -= j.groupby(level="pitch_type")["pred_" + name].transform("median")
    j["act"] = j["d_swstr"] * 100
    # 실제값 측정 오차 (두 시즌 표본 수로 계산한 표준오차)
    j["se"] = 100 * np.sqrt(j["swstr_a"] * (1 - j["swstr_a"]) / j["n_a"] + j["swstr_b"] * (1 - j["swstr_b"]) / j["n_b"])

    SUMMARY = []

    def report(d, title):
        print(f"\n=== {title}  (n={len(d)}, 실제값 평균 측정오차 ±{d['se'].mean():.1f}%p)")
        print(f"{'예측':14s} {'상관':>6s} {'95% 구간':>16s} {'기울기':>6s}  {'방향 일치(예측 |1%p| 이상)':>20s}")
        v_true = d["act"].var() - (d["se"] ** 2).mean()
        print(f"   (측정오차 때문에 완벽한 예측이어도 상관은 최대 약 {np.sqrt(max(v_true, 0) / d['act'].var()):.2f})")
        for name, lab in (("P0", "P0 릴리스만"), ("P1", "P1 +학습 무브"), ("P2", "P2 +실제 무브"),
                          ("P3", "P3 무브만(릴리스X)"), ("P4", "P4 학습무브만"), ("P5", "P5 학습무브+진입각")):
            x, y = d["pred_" + name].to_numpy(), d["act"].to_numpy()
            r, ci = corr_ci(x, y)
            slope = np.polyfit(x, y, 1)[0] if x.std() > 0 else np.nan
            big = np.abs(x) >= 1
            hit = (np.sign(x[big]) == np.sign(y[big])).mean() if big.sum() else np.nan
            print(f"{lab:14s} {r:+6.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}] {slope:+6.2f}   {hit:5.0%} ({big.sum()}개)")
            SUMMARY.append({"subset": title, "method": name, "label": lab, "n": len(d), "r": r, "lo": ci[0],
                            "hi": ci[1], "slope": slope,
                            "ceiling": np.sqrt(max(v_true, 0) / d["act"].var())})
    # 릴리스→무브 관계 자체가 맞았나: 2024에서 배운 관계로 예측한 무브 변화 vs 실제 무브 변화
    mvd = j[j["moved"]].reset_index()
    pv = np.array([MC.move_delta(move24, t, a_, b_, c_)["ivb"] for t, a_, b_, c_ in
                   zip(mvd["pitch_type"], mvd["d_rel_z"], mvd["d_rel_side"], mvd["d_ext"])])
    ph_ = np.array([MC.move_delta(move24, t, a_, b_, c_)["hb"] for t, a_, b_, c_ in
                    zip(mvd["pitch_type"], mvd["d_rel_z"], mvd["d_rel_side"], mvd["d_ext"])])
    print(f"릴리스→무브 관계 검증 (바꾼 {len(mvd)}개): 수직무브 상관 {np.corrcoef(pv, mvd['d_ivb'])[0, 1]:+.2f}, "
          f"가로무브 상관 {np.corrcoef(ph_, mvd['d_hb'])[0, 1]:+.2f}, "
          f"예측 크기 평균 {np.abs(pv).mean():.2f}in vs 실제 {mvd['d_ivb'].abs().mean():.2f}in")
    report(j, "전체")
    report(j[j["moved"]], f"릴리스를 {CHANGE * 30.48:.0f}cm 이상 바꾼 경우")
    report(j[j["moved"] & ~j["traded"]], "바꾼 경우 중 팀 그대로 (구장 측정 차이 제외)")
    for g, lab in (("fb", "패스트볼"), ("brk", "변화구"), ("off", "오프스피드")):
        d = j[j["moved"] & (j.index.get_level_values("pitch_type").map(RM.GROUP) == g)]
        if len(d) >= 15:
            report(d, f"바꾼 경우 · {lab}")

    show = j[j["moved"]].copy()
    show["big"] = show["pred_P1"].abs()
    show = show.sort_values("big", ascending=False).head(12).reset_index()
    names = P["a"].drop_duplicates("pitcher").set_index("pitcher")["name"]
    print("\n예측 변화가 컸던 사례 (단위 %p, 릴리스 cm)")
    for _, r in show.iterrows():
        print(f"  {names.get(r['pitcher'], r['pitcher'])!s:22.22s} {r['pitch_type']:3s} 타점{r['d_rel_z'] * 30.48:+5.0f} 옆{r['d_rel_side'] * 30.48:+5.0f} "
              f"익스{r['d_ext'] * 30.48:+5.0f} | 예측 P0 {r['pred_P0']:+5.1f} P1 {r['pred_P1']:+5.1f} P2 {r['pred_P2']:+5.1f} "
              f"| 실제 {r['act']:+5.1f} (±{r['se']:.1f})")
    j.reset_index().to_csv("val_release_result.csv", index=False)
    pd.DataFrame(SUMMARY).to_csv("val_release_summary.csv", index=False)
    print("\n저장: val_release_result.csv")
