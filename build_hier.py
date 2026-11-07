"""
계층 베이즈 모델 — 타자·투수·타자×구종 계열 효과를 한 모델에서 같이 추정하고, 선수마다 불확실성(구간)을 붙인다

지금까지의 방식: 타자 프로필(구간별 성적)을 단계별로 리그 평균 쪽에 축소 → LightGBM에 넣음. 축소 강도는 사람이 정했고,
                 "이 타자 슬라이더 헛스윙이 리그보다 6%p 높다"가 얼마나 확실한지는 알 수 없었다.
이 모델:  logit P(y) = 공 정보 GBM(선수 정보 없음, 오프셋) + 타자 u_b + 투수 v_p + 타자×구종 계열 w_bg
          u ~ N(0, τu²), v ~ N(0, τv²), w ~ N(0, τw²)  ← 분산(축소 강도)을 데이터에서 추정 (경험적 베이즈, 라플라스 근사)
          선수 효과마다 사후 평균·분산 → 구간
대상:     whiff = 스윙했을 때 헛스윙, swing = 스윙 여부
검증:     2024로 학습 → 2025 ① 투구 단위 정확도 (지금 모델 LightGBM+프로필, 신경망 임베딩과 비교, 표본 크기별)
                          ② 구간이 정직한가: 타자×구종 계열 2025 헛스윙률이 90% 구간에 실제로 90% 들어가나
                             (모수 불확실성 무시한 구간, 해마다 실력 변화까지 넣은 구간과 비교)

실행: python build_hier.py fit    (2024 학습, 10~20분)
      python build_hier.py val    (2025 검증)
      python build_hier.py save   (2024+2025로 최종 → hier_model.pkl, 앱에서 사용)
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

import build_matchup as B
import build_swing as W

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
OFF_F = B.PITCH_F + B.RELEASE_F + B.SIT_F + W.PREV_ALL          # 선수 정보 없는 공·상황 피처만
TARGETS = {"whiff": "swing", "swing": None}                     # 대상: (조건 열)
GRP = {0: "패스트볼", 1: "변화구", 2: "오프스피드"}


def load(year):
    d = B.load(FILES[year])
    d["grp"] = d["grp"].astype(int).clip(0, 2)
    return d


def subset(d, target):
    cond = TARGETS[target]
    return d[d[cond] == 1] if cond else d


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def fit_offset(d, target, n=300):
    m = W.clf(n).fit(d[OFF_F], d[target])
    return m


def oof_offset(d, target, n=300):
    """2-fold (경기 단위) 밖 예측 → 학습 데이터 오프셋이 과적합되지 않게"""
    fold = (d["game_pk"] % 2).to_numpy()
    o = np.zeros(len(d))
    for k in (0, 1):
        m = fit_offset(d[fold != k], target, n)
        o[fold == k] = logit(m.predict_proba(d[fold == k][OFF_F])[:, 1])
    return o


class RE:
    """랜덤 효과 하나 (레벨별 사후 평균·분산 + 분산 성분 τ²)"""

    def __init__(self, keys):
        self.keys = keys                              # 레벨 키 배열
        self.index = {k: i for i, k in enumerate(keys)}
        self.mu = np.zeros(len(keys))
        self.var = np.ones(len(keys))
        self.tau2 = 0.05
        self.n = np.zeros(len(keys))

    def idx(self, k_series):
        return pd.Series(k_series).map(self.index).fillna(-1).astype(int).to_numpy()

    def take(self, ix):
        return np.where(ix >= 0, self.mu[np.clip(ix, 0, None)], 0.0)


def fit_re(y, o, groups, outer=15, inner=3, log=print):
    """groups: {name: 키 배열}. 백피팅 + 뉴턴 + 분산 성분 EM (라플라스 근사)"""
    res = {}
    for name, keys in groups.items():
        uk, inv = np.unique(keys, return_inverse=True)
        r = RE(uk)
        r.ix = inv
        r.n = np.bincount(inv, minlength=len(uk)).astype(float)
        res[name] = r
    eta = o.copy()
    for it in range(outer):
        for name, r in res.items():
            eta -= r.mu[r.ix]
            for _ in range(inner):
                p = sig(eta + r.mu[r.ix])
                g = np.bincount(r.ix, y - p, len(r.keys))
                h = np.bincount(r.ix, p * (1 - p), len(r.keys))
                r.mu += (g - r.mu / r.tau2) / (h + 1 / r.tau2)
            p = sig(eta + r.mu[r.ix])
            h = np.bincount(r.ix, p * (1 - p), len(r.keys))
            r.var = 1 / (h + 1 / r.tau2)
            r.tau2 = float(np.mean(r.mu ** 2 + r.var))
            eta += r.mu[r.ix]
        if log and (it % 5 == 4 or it == outer - 1):
            log("    반복 %2d: " % (it + 1) + " · ".join(f"{k} τ={r.tau2 ** .5:.3f}" for k, r in res.items()))
    for r in res.values():
        del r.ix
    return res


def keys_of(d):
    return {"batter": d["batter"].to_numpy(), "pitcher": d["pitcher"].to_numpy(),
            "bg": (d["batter"].astype(np.int64) * 10 + d["grp"]).to_numpy()}


def predict_eta(res, o, d):
    k = keys_of(d)
    return o + sum(r.take(r.idx(k[name])) for name, r in res.items())


def fit_all(d, log=print):
    out = {}
    for target in TARGETS:
        s = subset(d, target)
        t0 = time.time()
        log(f"  [{target}] 투구 {len(s):,}개 — 공 정보 GBM 오프셋 (2-fold)")
        o = oof_offset(s, target)
        log(f"    오프셋 완료 ({time.time() - t0:.0f}초) → 랜덤 효과")
        res = fit_re(s[target].to_numpy(float), o, keys_of(s), log=log)
        m = fit_offset(s, target)                     # 새 시즌 예측용: 전체로 다시 학습
        out[target] = {"offset": m, "re": res}
        log(f"    완료 ({time.time() - t0:.0f}초)")
    return out


# ------------------------------------------------------------------ 검증
def coverage(res, o, d, target, level=0.9, drift=0.0, n_draw=400, seed=0, min_n=50):
    """타자×구종 계열(bg)마다: 2025 실제 비율이 예측 구간에 들어가나.
       구간 = 선수 효과의 사후 불확실성(+실력 변화 drift) + 이항 표본 오차"""
    rng = np.random.default_rng(seed)
    k = keys_of(d)
    rb, rp, rw = res["batter"], res["pitcher"], res["bg"]
    ib, ip, iw = rb.idx(k["batter"]), rp.idx(k["pitcher"]), rw.idx(k["bg"])
    base = o + rp.take(ip)                            # 투수 효과는 사후 평균으로 고정
    df = pd.DataFrame({"bg": k["bg"], "ib": ib, "iw": iw, "base": base, "y": d[target].to_numpy()})
    rows = []
    for bg, g in df.groupby("bg"):
        if len(g) < min_n:
            continue
        i_b, i_w = g["ib"].iloc[0], g["iw"].iloc[0]
        mb, vb = (rb.mu[i_b], rb.var[i_b]) if i_b >= 0 else (0.0, rb.tau2)
        mw, vw = (rw.mu[i_w], rw.var[i_w]) if i_w >= 0 else (0.0, rw.tau2)
        point = sig(g["base"].to_numpy() + mb + mw).mean()
        z = rng.normal(mb, np.sqrt(vb + drift), n_draw) + rng.normal(mw, np.sqrt(vw + drift), n_draw)
        rate = sig(g["base"].to_numpy()[None, :] + z[:, None]).mean(1)
        sim = rng.binomial(len(g), rate) / len(g)                 # 모수 불확실성 + 이항
        naive = rng.binomial(len(g), point, n_draw) / len(g)       # 이항만 (점 추정이 정확하다고 가정)
        a = (1 - level) / 2
        obs = g["y"].mean()
        lo, hi = np.quantile(sim, [a, 1 - a])
        nlo, nhi = np.quantile(naive, [a, 1 - a])
        rows.append({"bg": bg, "n": len(g), "obs": obs, "point": point, "lo": lo, "hi": hi,
                     "in": lo <= obs <= hi, "in_naive": nlo <= obs <= nhi, "known": i_w >= 0})
    return pd.DataFrame(rows)


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return roc_auc_score(y, p), log_loss(y, p)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if step == "fit":
        d24 = load(2024)
        M = fit_all(d24)
        pickle.dump(M, open("hier_2024.pkl", "wb"))
        print("저장: hier_2024.pkl")

    if step == "val":
        M = pickle.load(open("hier_2024.pkl", "rb"))
        d24, d25 = load(2024), load(2025)
        n24 = d24["batter"].value_counts()
        rows, cov_rows = [], []
        S = pickle.load(open("swing_model.pkl", "rb"))
        prof24 = B.batter_profile(d24)
        for target in TARGETS:
            s = subset(d25, target).copy()
            o = logit(M[target]["offset"].predict_proba(s[OFF_F])[:, 1])
            res = M[target]["re"]
            s["p_off"] = sig(o)
            s["p_hb"] = sig(predict_eta(res, o, s))
            # 비교: 지금 앱의 LightGBM+프로필 (swing_model.pkl), 신경망 임베딩 (2024 전용)
            te = B.attach(s, prof24)
            s["p_lgb"] = S[target].predict_proba(te[S["features"]])[:, 1]
            try:
                from build_embed import np_forward, prep
                z = np.load("embed_val2024.npz")
                Wn = {kk: z[kk] for kk in z.files}
                X, _ = prep(s, {"mu": Wn["mu"], "sd": Wn["sd"]})
                mp = lambda col, kk: s[col].map(dict(zip(Wn[kk + "_ids"].tolist(), Wn[kk + "_idx"].tolist()))).fillna(0).astype(int).to_numpy()
                ps, pw, _ = np_forward(Wn, X, mp("batter", "b"), mp("pitcher", "p"))
                s["p_nn"] = ps if target == "swing" else pw
            except Exception as e:
                print("신경망 비교 생략:", e)
                s["p_nn"] = np.nan
            s["n24"] = s["batter"].map(n24).fillna(0).to_numpy()
            buckets = [("전체", s), ("2024 표본 0 (신인)", s[s["n24"] == 0]), ("1~500구", s[(s["n24"] > 0) & (s["n24"] <= 500)]),
                       ("500~1500구", s[(s["n24"] > 500) & (s["n24"] <= 1500)]), ("1500구+", s[s["n24"] > 1500])]
            print(f"\n[{target}] 2024 학습 → 2025 예측  AUC / logloss")
            for lab, g in buckets:
                line = f"  {lab:14s} n={len(g):>7,} |"
                for nm, col in (("공 정보만", "p_off"), ("계층 베이즈", "p_hb"), ("LightGBM+프로필", "p_lgb"), ("신경망", "p_nn"),
                                ("계층+신경망 평균", None)):
                    pr = (g["p_hb"] + g["p_nn"]) / 2 if col is None else g[col]
                    if pr.isna().all():
                        continue
                    a1, l1 = metrics(g[target], pr)
                    line += f" {nm} {a1:.4f}/{l1:.4f} |"
                    rows.append({"target": target, "bucket": lab, "n": len(g), "model": nm, "auc": a1, "ll": l1})
                print(line)
            # ② 구간 정직성
            for drift_lab, dr in (("실력 변화 없음", 0.0), ("실력 변화 포함", None)):
                if dr is None:     # 해마다 실력 변화: 2024 전반기 → 후반기 차이로 추정 (2025는 안 씀)
                    dr = M[target].get("drift", 0.0)
                cv = coverage(res, o, s, target, drift=dr)
                for lab, g in (("전체", cv), ("2024에 있던 타자", cv[cv["known"]]), ("신인", cv[~cv["known"]])):
                    if len(g) == 0:
                        continue
                    print(f"  90% 구간 포함률 [{drift_lab}] {lab:10s} 타자×계열 {len(g):>4}개: 계층 베이즈 {g['in'].mean():.0%}"
                          f" | 점 추정+이항만 {g['in_naive'].mean():.0%} | 구간 폭 중앙값 {(g['hi'] - g['lo']).median() * 100:.1f}%p")
                    cov_rows.append({"target": target, "drift": drift_lab, "group": lab, "n": len(g), "cover_hb": g["in"].mean(),
                                     "cover_naive": g["in_naive"].mean(), "width": (g["hi"] - g["lo"]).median()})
        pd.DataFrame(rows).to_csv("val_hier_summary.csv", index=False)
        pd.DataFrame(cov_rows).to_csv("val_hier_coverage.csv", index=False)
        print("저장: val_hier_summary.csv, val_hier_coverage.csv")

    if step == "drift":
        # 해마다 실력 변화(drift): 2024 전반기로 학습한 효과 vs 후반기로 학습한 효과의 차이 분산 − 두 사후 분산 → 2025는 안 씀
        d24 = load(2024)
        M = pickle.load(open("hier_2024.pkl", "rb"))
        a, b = d24[d24["game_date"] < "2024-07-01"], d24[d24["game_date"] >= "2024-07-01"]
        for target in TARGETS:
            ra, rb_ = {}, {}
            for half, dd, store in (("전반기", a, ra), ("후반기", b, rb_)):
                s = subset(dd, target)
                o = logit(M[target]["offset"].predict_proba(s[OFF_F])[:, 1])
                store.update(fit_re(s[target].to_numpy(float), o, keys_of(s), outer=10, log=None))
            x = pd.DataFrame({"m": ra["bg"].mu, "v": ra["bg"].var}, index=ra["bg"].keys).join(
                pd.DataFrame({"m2": rb_["bg"].mu, "v2": rb_["bg"].var}, index=rb_["bg"].keys), how="inner")
            xb = pd.DataFrame({"m": ra["batter"].mu, "v": ra["batter"].var}, index=ra["batter"].keys).join(
                pd.DataFrame({"m2": rb_["batter"].mu, "v2": rb_["batter"].var}, index=rb_["batter"].keys), how="inner")
            dv = max(np.var(x["m"] - x["m2"]) - (x["v"] + x["v2"]).mean(), 0) / 2 \
                + max(np.var(xb["m"] - xb["m2"]) - (xb["v"] + xb["v2"]).mean(), 0) / 2
            # 반 시즌 간 변화 → 한 시즌 뒤는 대략 두 배 거리로 보고 그대로 사용 (보수적으로 같은 값)
            M[target]["drift"] = float(dv)
            print(f"  [{target}] 실력 변화 분산 (효과 하나당) {dv:.4f} (표준편차 {dv ** .5:.3f})")
        pickle.dump(M, open("hier_2024.pkl", "wb"))

    if step == "save":
        both = pd.concat([load(2024), load(2025)], ignore_index=True)
        M = fit_all(both)
        # 앱용: 타자×계열별 헛스윙·스윙 효과 (사후 평균, 표준편차) + 리그 기준 확률로 바꾼 값
        out = {}
        for target in TARGETS:
            s = subset(both, target)
            res = M[target]["re"]
            base = sig(logit(M[target]["offset"].predict_proba(s[OFF_F])[:, 1]))
            lg = s.assign(p=base).groupby("grp")["p"].mean()            # 계열별 리그 기준 확률 (공 정보 모델 평균)
            rw, rb = res["bg"], res["batter"]
            t = pd.DataFrame({"bg": rw.keys, "mu_w": rw.mu, "var_w": rw.var, "n": rw.n})
            t["batter"], t["grp"] = t["bg"] // 10, t["bg"] % 10
            bmu = dict(zip(rb.keys, rb.mu))
            bvar = dict(zip(rb.keys, rb.var))
            t["mu"] = t["mu_w"] + t["batter"].map(bmu)
            t["sd"] = np.sqrt(t["var_w"] + t["batter"].map(bvar))
            # 이 타자가 실제로 스윙한(받은) 공에서의 비율과 90% 구간 — 2025 검증(포함률)에서 확인한 바로 그 값
            o = logit(base) + res["pitcher"].take(res["pitcher"].idx(s["pitcher"].to_numpy()))
            bg = (s["batter"].astype(np.int64) * 10 + s["grp"]).to_numpy()
            tt = t.set_index("bg")
            m_, s_ = tt["mu"].reindex(bg).to_numpy(), tt["sd"].reindex(bg).to_numpy()
            q = pd.DataFrame({"bg": bg, "r": sig(o + m_), "lo": sig(o + m_ - 1.645 * s_), "hi": sig(o + m_ + 1.645 * s_)})
            agg = q.groupby("bg")[["r", "lo", "hi"]].mean()
            t = t.join(agg, on="bg")
            obs = s.groupby("grp")[target].mean()                      # 리그 실제 비율 (비교 기준)
            out[target] = {"table": t[["batter", "grp", "n", "mu", "sd", "r", "lo", "hi"]], "league": lg.to_dict(),
                           "league_obs": obs.to_dict(),
                           "tau": {k: r.tau2 ** .5 for k, r in res.items()},
                           "pitcher": pd.DataFrame({"pitcher": res["pitcher"].keys, "mu": res["pitcher"].mu,
                                                    "sd": np.sqrt(res["pitcher"].var), "n": res["pitcher"].n})}
        pickle.dump(out, open("hier_model.pkl", "wb"))
        print("저장: hier_model.pkl")
