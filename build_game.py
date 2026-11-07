"""
볼배합 게임이론 — '좋은 공만 던지면 타자가 노린다'를 데이터로 재고, 섞어 던지기 최적 비율 계산

1) 착취(exploitation) 측정:  어떤 카운트에서 투수가 한 구종을 평소보다 많이 던질수록(예측 가능할수록)
   그 구종의 결과가 모델 예상보다 나빠지나?
     잔차 = 실제 실점 가치 − 매치업 모델 예상 (공 질·코스·카운트·타자 성향 반영된 예상)
     Δ구사율 = 그 카운트에서의 그 구종 비율 − 그 투수의 전체 비율
     투수×구종 고정효과(같은 투수·같은 구종끼리 비교) 회귀:  잔차 ~ β · Δ구사율
   β > 0 이면 '많이 던질수록 손해' = 타자가 노리고 들어온다는 증거.
   ※ 투수는 원래 그 카운트에서 잘 통하는 구종을 더 던지므로(선택 편향), 실제 착취는 이 β보다 더 클 수 있음.
   2024로 추정 → 2025에서 같은 크기로 재현되는지 확인. 스윙·헛스윙 반응도 같이 (타자가 노리면 더 휘두르고 덜 헛친다)

2) 섞어 던지기 최적화 (game_mix):  구종 x의 기대 실점 = 모델 값 + β·(구사율 − 평소 구사율)
   → 기대 실점 합이 최소가 되는 비율 (내쉬 균형과 같은 '한계 가치 같게 만들기' 조건, 수식으로 바로 풂)

실행: python build_game.py exploit    (2~4분)
"""
import pickle
import sys

import numpy as np
import pandas as pd

import build_matchup as B

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
MIN_P, MIN_CELL = 500, 25


def usage_frame(d):
    d = d.copy()
    d["count"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    n_p = d.groupby("pitcher")["pitch_type"].transform("size")
    d = d[n_p >= MIN_P]
    u_all = d.groupby(["pitcher", "pitch_type"]).size() / d.groupby("pitcher").size()
    cell = d.groupby(["pitcher", "count"]).size()
    u_cnt = d.groupby(["pitcher", "count", "pitch_type"]).size() / cell
    d["u_all"] = pd.MultiIndex.from_frame(d[["pitcher", "pitch_type"]]).map(u_all)
    d["u_cnt"] = pd.MultiIndex.from_frame(d[["pitcher", "count", "pitch_type"]]).map(u_cnt)
    d["cell_n"] = pd.MultiIndex.from_frame(d[["pitcher", "count"]]).map(cell)
    d = d[d["cell_n"] >= MIN_CELL]
    d["dU"] = d["u_cnt"] - d["u_all"]
    return d


def fe_beta(d, y, w=None, B_=300, seed=0):
    """투수×구종 고정효과 회귀 기울기 + 투수 단위 부트스트랩 95% 구간"""
    key = d["pitcher"].astype(str) + "_" + d["pitch_type"]
    yy = d[y] - d.groupby(key)[y].transform("mean")
    xx = d["dU"] - d.groupby(key)["dU"].transform("mean")
    b = float((xx * yy).sum() / (xx ** 2).sum())
    rng = np.random.default_rng(seed)
    ps = d["pitcher"].unique()
    g = pd.DataFrame({"p": d["pitcher"].values, "xy": (xx * yy).values, "xx": (xx ** 2).values}).groupby("p").sum()
    bs = []
    for _ in range(B_):
        s = g.loc[rng.choice(ps, len(ps))]
        bs.append(s["xy"].sum() / s["xx"].sum())
    return b, np.percentile(bs, [2.5, 97.5])


def game_mix(rv0, u_ref, beta, floor=0.0):
    """기대 실점 Σ u_x (rv0_x + β(u_x − u_ref_x)) 최소화, Σu=1, u≥floor
    1차 조건: rv0_x + β(2u_x − u_ref_x) = λ  →  u_x = (λ − rv0_x + β u_ref_x) / (2β)  (물 채우기 방식으로 λ 찾기)"""
    rv0, u_ref = np.asarray(rv0, float), np.asarray(u_ref, float)
    if beta <= 0:
        u = np.zeros_like(rv0)
        u[np.argmin(rv0)] = 1
        return u
    lo, hi = (rv0 - beta * u_ref).min() - 10, (rv0 - beta * u_ref).max() + 10 + 2 * beta
    for _ in range(100):
        lam = (lo + hi) / 2
        u = np.clip((lam - rv0 + beta * u_ref) / (2 * beta), floor, 1)
        lo, hi = (lam, hi) if u.sum() < 1 else (lo, lam)
    return u / u.sum()


def expected(u, rv0, u_ref, beta):
    u, rv0, u_ref = map(lambda a: np.asarray(a, float), (u, rv0, u_ref))
    return float((u * (rv0 + beta * (u - u_ref))).sum())


def cells(d, min_cell=40, min_u=0.03):
    """투수×카운트 칸마다: 구종별 모델 기대 실점(rv0, 100구당), 평소 구사율, 그 카운트 실제 구사율"""
    d = d.copy()
    d["count"] = d["balls"].astype(int).astype(str) + "-" + d["strikes"].astype(int).astype(str)
    d = d[d.groupby("pitcher")["pitch_type"].transform("size") >= MIN_P]
    u_all = d.groupby(["pitcher", "pitch_type"]).size() / d.groupby("pitcher").size()
    rv_t = d.groupby(["pitcher", "pitch_type"])["rv_hat"].mean() * 100            # 구종 전체 평균
    rv_c = d.groupby(["pitcher", "count"])["rv_hat"].mean() * 100                 # 그 카운트 평균
    rv_p = d.groupby("pitcher")["rv_hat"].mean() * 100
    rv_tc = d.groupby(["pitcher", "count", "pitch_type"])["rv_hat"].agg(["mean", "size"])
    out = []
    for (p, c), g in d.groupby(["pitcher", "count"]):
        if len(g) < min_cell:
            continue
        types = [t for t in u_all.loc[p].index if u_all.loc[(p, t)] >= min_u]
        if len(types) < 2:
            continue
        rv0, ua, uc = [], [], []
        for t in types:
            if (p, c, t) in rv_tc.index and rv_tc.loc[(p, c, t), "size"] >= 5:
                rv0.append(rv_tc.loc[(p, c, t), "mean"] * 100)
            else:                                              # 그 카운트에서 거의 안 던진 구종: 구종 평균 + 카운트 보정
                rv0.append(rv_t.loc[(p, t)] + rv_c.loc[(p, c)] - rv_p.loc[p])
            ua.append(u_all.loc[(p, t)])
            uc.append((g["pitch_type"] == t).mean())
        ua, uc = np.array(ua) / sum(ua), np.array(uc) / max(sum(uc), 1e-9)
        out.append({"pitcher": p, "count": c, "n": len(g), "types": types, "rv0": np.array(rv0), "u_all": ua, "u_cnt": uc})
    return out


def mix_loss(cs, beta):
    """실제 구사율과 예측 구사율의 교차 엔트로피 (투구 수 가중)"""
    tot, w = 0.0, 0
    for c in cs:
        u = np.clip(game_mix(c["rv0"], c["u_all"], beta), 0.005, 1)
        u = u / u.sum()
        tot += -(c["u_cnt"] * np.log(u)).sum() * c["n"]
        w += c["n"]
    return tot / w


def base_loss(cs, kind):
    tot, w = 0.0, 0
    for c in cs:
        if kind == "same":
            u = c["u_all"]
        elif kind == "greedy":
            u = np.full(len(c["rv0"]), 0.005)
            u[np.argmin(c["rv0"])] = 1
        u = np.clip(u, 0.005, 1)
        u = u / u.sum()
        tot += -(c["u_cnt"] * np.log(u)).sum() * c["n"]
        w += c["n"]
    return tot / w


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "exploit"
    M = pickle.load(open("matchup_model.pkl", "rb"))
    S = pickle.load(open("swing_model.pkl", "rb"))
    if step == "mix":
        # 역(逆) 게임이론: MLB 투수들의 실제 카운트별 구종 선택이 '타자가 얼마나 노린다고 보는지(β)'를 드러냄
        res = {}
        for yr in (2024, 2025):
            d = B.attach(B.load(FILES[yr]), M["profile"])
            d["rv_hat"] = M["model"].predict(d[M["features"]])
            res[yr] = cells(d)
            print(f"{yr}: 투수×카운트 칸 {len(res[yr]):,}개")
        grid = [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
        l24 = {b: mix_loss(res[2024], b) for b in grid}
        bstar = min(l24, key=l24.get)
        print("\n[2024에서 β 고르기] 교차 엔트로피 (낮을수록 실제 투수 선택과 비슷)")
        for b in grid:
            print(f"  β={b:>5}: {l24[b]:.4f}" + ("  ← 최적" if b == bstar else ""))
        print(f"  비교: 모든 카운트 같은 비율 {base_loss(res[2024], 'same'):.4f} / 제일 좋은 공만 {base_loss(res[2024], 'greedy'):.4f}")
        print(f"\n[2025에 그대로 적용] β={bstar}: {mix_loss(res[2025], bstar):.4f} / 같은 비율 {base_loss(res[2025], 'same'):.4f} "
              f"/ 제일 좋은 공만 {base_loss(res[2025], 'greedy'):.4f}")
        # 기대 실점 비교 (β 가정 아래): 제일 좋은 공만 vs 게임이론 섞기 vs 실제
        ex = {"greedy": [], "mix": [], "actual": []}
        for c in res[2025]:
            g = np.zeros(len(c["rv0"]))
            g[np.argmin(c["rv0"])] = 1
            for k, u in (("greedy", g), ("mix", game_mix(c["rv0"], c["u_all"], bstar)), ("actual", c["u_cnt"])):
                ex[k].append(expected(u, c["rv0"], c["u_all"], bstar) * c["n"])
        n = sum(c["n"] for c in res[2025])
        print("  β 가정 아래 기대 실점 (100구당): " + " / ".join(f"{k} {sum(v) / n:+.3f}" for k, v in ex.items()))
        with open("game_mix.pkl", "wb") as f:
            pickle.dump({"beta": bstar, "loss24": l24, "same24": base_loss(res[2024], "same"),
                         "greedy24": base_loss(res[2024], "greedy"), "loss25": mix_loss(res[2025], bstar),
                         "same25": base_loss(res[2025], "same"), "greedy25": base_loss(res[2025], "greedy"),
                         "exp25": {k: sum(v) / n for k, v in ex.items()}}, f)
        print("저장: game_mix.pkl")
        sys.exit()
    out = {}
    for yr in (2024, 2025):
        d = B.attach(B.load(FILES[yr]), M["profile"])
        d["rv_hat"] = M["model"].predict(d[M["features"]])
        d["resid"] = (d[B.TARGET] - d["rv_hat"]) * 100                     # 100구당
        Xs = d[S["features"]]
        d["p_sw"] = S["swing"].predict_proba(Xs)[:, 1]
        d["p_wh"] = S["whiff"].predict_proba(Xs)[:, 1]
        d["r_sw"] = d["swing"] - d["p_sw"]
        d["r_wh"] = np.where(d["swing"] == 1, d["whiff"] - d["p_wh"], np.nan)
        u = usage_frame(d)
        res = {}
        print(f"\n=== {yr}  (투수 {u['pitcher'].nunique()}명, 투구 {len(u):,}개, Δ구사율 표준편차 {u['dU'].std():.3f})")
        for y, lab, sc in (("resid", "실점 가치 잔차 (100구당)", 1), ("r_sw", "스윙 잔차 (%p)", 100), ("r_wh", "헛스윙 잔차 (%p, 스윙만)", 100)):
            dd = u.dropna(subset=[y])
            b, ci = fe_beta(dd, y)
            res[y] = (b, ci)
            print(f"  {lab:24s} β = {b * sc / 10:+.3f} per +10%p 구사율  [{ci[0] * sc / 10:+.3f}, {ci[1] * sc / 10:+.3f}]")
        # 구간별 평균 (그래프용)
        u["bin"] = pd.cut(u["dU"], [-1, -0.2, -0.1, -0.03, 0.03, 0.1, 0.2, 1],
                          labels=["≤-20%p", "-20~-10", "-10~-3", "±3", "+3~+10", "+10~+20", "≥+20%p"])
        key = u["pitcher"].astype(str) + "_" + u["pitch_type"]
        u["resid_w"] = u["resid"] - u.groupby(key)["resid"].transform("mean")
        tab = u.groupby("bin", observed=True).agg(n=("resid", "size"), resid=("resid_w", "mean"))
        print(tab.round(2).to_string())
        out[yr] = {"beta": {k: (float(v[0]), [float(x) for x in v[1]]) for k, v in res.items()},
                   "bins": tab.reset_index().assign(bin=lambda t: t["bin"].astype(str)).to_dict("list"),
                   "n": int(len(u)), "pitchers": int(u["pitcher"].nunique())}
    with open("game_exploit.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\n저장: game_exploit.pkl")
