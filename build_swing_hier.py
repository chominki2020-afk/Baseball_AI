"""
사례 9 — 타자 스윙 what-if를 '스윙 하나하나 + 타자 효과'로 다시 만들기

문제 (사례 2): 시즌 평균 스윙 값을 넣으면 그 값이 선수 이름표가 되어, 모델이 스윙의 효과가 아니라 선수를 외운다.
지금 앱(F): 타자 스타일을 계수 몇십 개짜리 매끄러운 식으로 → what-if는 되지만 타구 질 예측 상관 0.59 → 0.48
이 실험:
  G  스윙 단위 + 타자 효과:  y ~ 공 상황 GBM(오프셋) + 타자 효과 u_b(랜덤, 분산은 데이터에서) + 그 스윙의 실제 값으로 만든 매끄러운 식
     → "누구냐"는 u_b가 흡수하고, 스윙 값의 계수는 주로 같은 타자 안의 스윙 차이에서 배운다 (선수를 외울 수 없음)
  G0 G에서 타자 효과만 뺀 것 (타자 효과가 하는 일을 보려고)
  H  F의 시즌 평균 식 + 그 스윙의 평균과의 차이(선형) + 타자 효과 (타자 간 / 타자 안 차이를 나눠 추정)
  공통: 경쟁 스윙만 학습 (그 타자 배트 스피드 하위 10% = 체크·방어 스윙 제외, Savant 기준과 같은 생각)
검증: 사례 2와 똑같이 — 2024 학습, 두 시즌 모두 300스윙 이상인 타자 286명의 2024→2025 실제 스윙 변화를
      2024 스윙에 더해 예측한 결과 변화 vs 실제 결과 변화 (상관·기울기), 타자 유형별 어택 앵글 효과, 2025 투구 단위 정확도

실행: python build_swing_hier.py cache   (CSV → 스윙만 추려 저장, 한 번)
      python build_swing_hier.py fit     (2024 학습)
      python build_swing_hier.py val     (검증, 지금 앱 모델 F와 비교)
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

import build_batter_swing as BS
import build_hier as H

SW, PITCH_F, STYLE = BS.SW, BS.PITCH_F, BS.STYLE
PAIRS = BS.PAIRS
LOW_CUT = 0.10                     # 타자별 배트 스피드 하위 10% 스윙은 학습에서 제외
RIDGE = 1.0                        # 스윙 식 계수의 약한 L2 (표준화된 항 기준)


# ------------------------------------------------------------------ 데이터
def cache():
    keep = ["batter", "game_pk", "stand", "whiff", "contact", "in_play", "xwoba", "season"] + PITCH_F + SW + STYLE
    out = {}
    for y in (2024, 2025):
        s = BS.load(y)
        out[y] = s[[c for c in keep if c in s.columns]].reset_index(drop=True)
        print(y, len(out[y]))
    pickle.dump(out, open("swing_cache.pkl", "wb"))


def competitive(s):
    s = s.dropna(subset=SW)                                 # 컨택 위치가 비어 있는 스윙 소수(0.1%) 제외
    cut = s.groupby("batter")["bat_speed"].transform(lambda v: v.quantile(LOW_CUT))
    return s[s["bat_speed"] >= cut]


# ------------------------------------------------------------------ 식 (항 만들기)
def zterms(Z, X):
    """Z: 표준화된 스윙 값 (n×6). 1차·2차·공 높이/코스와의 상호작용·스윙 요소끼리 조합"""
    zn = X["z_norm"].to_numpy(float)[:, None] - 0.5
    xi = X["x_in"].to_numpy(float)[:, None]
    pr = np.stack([Z[:, i] * Z[:, j] for i, j in PAIRS], axis=1)
    return np.hstack([Z, Z ** 2, Z * zn, Z * xi, pr])


def design(kind, X, mu, sd):
    if kind in ("G", "G0"):                                   # 그 스윙의 실제 값
        return zterms(np.nan_to_num((X[SW].to_numpy(float) - mu) / sd), X)   # 빈 값 = 평균
    if kind == "H":                                           # 시즌 평균 식 + 평균과의 차이 (선형)
        zm = (X[STYLE].to_numpy(float) - mu) / sd
        dev = np.nan_to_num((X[SW].to_numpy(float) - X[STYLE].to_numpy(float)) / sd)
        return np.hstack([zterms(zm, X), dev])
    raise ValueError(kind)


# ------------------------------------------------------------------ 적합
def irls_logit(y, off, T, lam=RIDGE, iters=8, beta=None):
    n, k = T.shape
    A = np.hstack([np.ones((n, 1)), T])
    b = np.zeros(k + 1) if beta is None else beta.copy()
    P = np.eye(k + 1) * lam
    P[0, 0] = 0
    for _ in range(iters):
        p = H.sig(off + A @ b)
        w = p * (1 - p)
        g = A.T @ (y - p) - P @ b
        Hm = (A * w[:, None]).T @ A + P
        step = np.linalg.solve(Hm, g)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    return b


def ridge_gauss(y, off, T, lam=RIDGE):
    n, k = T.shape
    A = np.hstack([np.ones((n, 1)), T])
    P = np.eye(k + 1) * lam
    P[0, 0] = 0
    return np.linalg.solve(A.T @ A + P, A.T @ (y - off))


def re_gauss(r, keys, tau2=0.001, iters=30):
    """가우스 랜덤 절편: 사후 평균 = 합 / (n + σ²/τ²), τ²·σ²는 EM"""
    uk, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv, minlength=len(uk)).astype(float)
    s2 = r.var()
    for _ in range(iters):
        mu = np.bincount(inv, r, len(uk)) / (n + s2 / tau2)
        var = 1 / (n / s2 + 1 / tau2)
        tau2 = float(np.mean(mu ** 2 + var))
        s2 = float(np.mean((r - mu[inv]) ** 2) + np.mean(var[inv]))
    re = H.RE(uk)
    re.mu, re.var, re.tau2, re.n = mu, var, tau2, n
    return re


_OFF = {}


def offsets(s, target, family):
    """공 상황만 본 GBM: 2-fold(경기 단위) 밖 예측 = 학습용 오프셋, 전체 학습 = 예측용. 모델끼리 같은 데이터면 재사용"""
    key = (target, len(s), int(s["batter"].sum() % 1000003))
    if key in _OFF:
        return _OFF[key]
    fold = (s["game_pk"] % 2).to_numpy()
    o = np.zeros(len(s))
    for k in (0, 1):
        tr, te = s[fold != k], s[fold == k]
        if family == "binomial":
            m = BS.clf().fit(tr[PITCH_F], tr[target])
            o[fold == k] = H.logit(m.predict_proba(te[PITCH_F])[:, 1])
        else:
            m = BS.reg().fit(tr[PITCH_F], tr[target])
            o[fold == k] = m.predict(te[PITCH_F])
    full = (BS.clf() if family == "binomial" else BS.reg()).fit(s[PITCH_F], s[target])
    _OFF[key] = (o, full)
    return _OFF[key]


def fit_part(kind, s, target, family, T, log=print):
    """오프셋(2-fold 밖 GBM) + 스윙 식 + (G·H) 타자 랜덤 절편, 번갈아 적합"""
    y = s[target].to_numpy(float)
    t0 = time.time()
    o, full = offsets(s, target, family)
    keys = s["batter"].to_numpy()
    u = np.zeros(len(s))
    re, beta = None, None
    rounds = 1 if kind == "G0" else 5
    for it in range(rounds):
        if family == "binomial":
            beta = irls_logit(y, o + u, T, beta=beta)
            if kind != "G0":
                re = H.fit_re(y, o + np.hstack([np.ones((len(s), 1)), T]) @ beta, {"batter": keys}, outer=6,
                              log=None)["batter"]
        else:
            beta = ridge_gauss(y, o + u, T)
            if kind != "G0":
                fx = o + np.hstack([np.ones((len(s), 1)), T]) @ beta
                re = re_gauss(y - fx, keys)
        if re is not None:
            u = re.take(re.idx(keys))
    log(f"    [{kind}·{target}] {len(s):,}개, {time.time() - t0:.0f}초" +
        (f", 타자 효과 τ={re.tau2 ** .5:.3f}" if re is not None else ""))
    return {"offset": full, "beta": beta, "re": re, "family": family}


def fit_model(kind, s, log=print):
    s = competitive(s)
    if kind == "H":
        per = s.groupby("batter")[STYLE].first()
        mu, sd = per.mean().to_numpy(), per.std().to_numpy()
    else:
        mu, sd = s[SW].mean().to_numpy(), s[SW].std().to_numpy()
    m = {"kind": kind, "mu": mu, "sd": sd}
    con = s[s["contact"] == 1]
    b = con[(con["in_play"] == 1) & con["xwoba"].notna()]
    for part, d, tgt, fam in (("miss", s, "whiff", "binomial"), ("bip", con, "in_play", "binomial"),
                              ("xwcon", b, "xwoba", "gauss")):
        m[part] = fit_part(kind, d, tgt, fam, design(kind, d, mu, sd), log)
    return m


def predict(m, X, use_re=True):
    if m.get("kind") not in ("G", "G0", "H"):
        return BS.predict(m, X)
    T = design(m["kind"], X, m["mu"], m["sd"])
    A = np.hstack([np.ones((len(X), 1)), T])
    out = {}
    for part in ("miss", "bip", "xwcon"):
        p = m[part]
        if p["family"] == "binomial":
            eta = H.logit(p["offset"].predict_proba(X[PITCH_F])[:, 1]) + A @ p["beta"]
        else:
            eta = p["offset"].predict(X[PITCH_F]) + A @ p["beta"]
        if use_re and p["re"] is not None:
            eta = eta + p["re"].take(p["re"].idx(X["batter"].to_numpy()))
        out[part] = H.sig(eta) if p["family"] == "binomial" else eta
    df = pd.DataFrame(out, index=X.index)
    df["xw_sw"] = (1 - df["miss"]) * df["bip"] * df["xwcon"]
    return df


# ------------------------------------------------------------------ 실행
if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if step == "cache":
        cache()

    if step == "fit":
        C = pickle.load(open("swing_cache.pkl", "rb"))
        s24 = C[2024]
        models = {}
        for kind in sys.argv[2:] or ["G", "G0", "H"]:
            print(f"== {kind}")
            models[kind] = fit_model(kind, s24)
        try:
            old = pickle.load(open("swing_hier_2024.pkl", "rb"))
            old.update(models)
            models = old
        except FileNotFoundError:
            pass
        pickle.dump(models, open("swing_hier_2024.pkl", "wb"))
        print("저장: swing_hier_2024.pkl", list(models))

    if step == "val":
        C = pickle.load(open("swing_cache.pkl", "rb"))
        s24, s25 = C[2024], C[2025]
        M = pickle.load(open("swing_hier_2024.pkl", "rb"))
        old = pickle.load(open("batter_val_models.pkl", "rb"))
        M = {"F (지금 앱)": old["F"], "B (처음, 선수 암기)": old["B"], **M}
        # ---- 사례 2와 같은 286명, 같은 샘플
        a, b = BS.batter_summary(s24), BS.batter_summary(s25)
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["n_a"] >= BS.MIN_SW) & (j["n_b"] >= BS.MIN_SW)]
        for c in SW + ["whiff", "xwcon", "xw_sw", "bip_rate"]:
            j["d_" + c] = j[c + "_b"] - j[c + "_a"]
            j["d_" + c] -= j["d_" + c].median()
        print(f"검증 타자 {len(j)}명")
        smp = s24[s24["batter"].isin(j.index)].groupby("batter", group_keys=False).sample(frac=1, random_state=0)
        smp = smp.groupby("batter").head(250)
        shift = j[["d_" + c for c in SW]].rename(columns=lambda c: c[2:])
        x = smp.copy()
        for c in SW:
            add = x["batter"].map(shift[c])
            x[c] = x[c] + add
            x["m_" + c] = x["m_" + c] + add
        deltas = {}
        for nm, m in M.items():
            d = (predict(m, x) - predict(m, smp)).groupby(smp["batter"]).mean()
            deltas[nm] = d - d.median()
        rows = []
        for tgt, lab, col in (("whiff", "헛스윙%(스윙당)", "miss"), ("xwcon", "인플레이 xwOBA (타구 질)", "xwcon"),
                              ("xw_sw", "스윙당 xwOBA", "xw_sw"), ("bip_rate", "맞혔을 때 인플레이%", "bip")):
            act = j["d_" + tgt].to_numpy()
            print(f"\n=== {lab}  (예측한 변화 vs 실제 변화, {len(j)}명)")
            for nm, d in deltas.items():
                p = d[col].reindex(j.index).to_numpy()
                r, ci = BS.corr_ci(p, act)
                slope = np.polyfit(p, act, 1)[0]
                print(f"  {nm:18s} 상관 {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  기울기 {slope:+.2f}")
                rows.append({"target": tgt, "label": lab, "model": nm, "r": r, "lo": ci[0], "hi": ci[1], "slope": slope})
        # ---- 차이의 구간: 같은 타자 부트스트랩으로 (G − F) 상관 차이
        rng = np.random.default_rng(1)
        print("\n[상관 차이, 같은 타자 부트스트랩 95% 구간]")
        for tgt, col in (("whiff", "miss"), ("xwcon", "xwcon"), ("xw_sw", "xw_sw")):
            act = j["d_" + tgt].to_numpy()
            for a_, b_ in (("G", "F (지금 앱)"), ("H", "F (지금 앱)"), ("G", "G0")):
                if a_ not in deltas or b_ not in deltas:
                    continue
                pa, pb = deltas[a_][col].reindex(j.index).to_numpy(), deltas[b_][col].reindex(j.index).to_numpy()
                diffs = []
                for _ in range(2000):
                    i = rng.integers(0, len(act), len(act))
                    diffs.append(np.corrcoef(pa[i], act[i])[0, 1] - np.corrcoef(pb[i], act[i])[0, 1])
                d0 = np.corrcoef(pa, act)[0, 1] - np.corrcoef(pb, act)[0, 1]
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                print(f"  {tgt:6s} {a_} − {b_}: {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                rows.append({"target": tgt, "label": "diff", "model": f"{a_} - {b_}", "r": d0, "lo": lo, "hi": hi})
        # ---- 타자 유형별 어택 앵글 +1°
        j["grp"] = pd.qcut(j["bat_speed_a"], 3, labels=["컨택형", "중간", "파워형"])
        sm2 = smp.copy()
        sm2["attack_angle"] += 1
        sm2["m_attack_angle"] += 1
        eff = {nm: (predict(m, sm2) - predict(m, smp)).groupby(smp["batter"]).mean() for nm, m in M.items()}
        print("\n[타자 유형별] 어택 앵글 +1° — 실제(시즌 간 변화 회귀) vs 모델  (타구 질 xwOBA 포인트)")
        grow = []
        rng = np.random.default_rng(0)
        for g, d in j.groupby("grp", observed=True):
            xx, yy = d["d_attack_angle"].to_numpy(), d["d_xwcon"].to_numpy()
            bb = np.polyfit(xx, yy, 1)[0] * 1000
            bs = [np.polyfit(xx[i], yy[i], 1)[0] * 1000 for i in (rng.integers(0, len(xx), len(xx)) for _ in range(1000))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line = f"  {g:5s} 실제 {bb:+.1f} [{lo:+.1f}, {hi:+.1f}] |"
            row = {"group": str(g), "actual": bb, "lo": lo, "hi": hi}
            for nm, e in eff.items():
                v = e["xwcon"].reindex(d.index).mean() * 1000
                line += f" {nm.split()[0]} {v:+.1f}"
                row[nm] = v
            print(line)
            grow.append(row)
        # ---- 2025 투구(스윙) 단위 정확도: 2024 모델로 2025 스윙 예측
        from sklearn.metrics import log_loss, roc_auc_score
        print("\n[2025 스윙 단위 예측 정확도] 헛스윙 AUC / logloss · 타구 질 R²")
        t25 = s25.sample(n=min(120000, len(s25)), random_state=0)
        acc = []
        for nm, m in M.items():
            p = predict(m, t25)
            pm = np.clip(p["miss"].to_numpy(), 1e-5, 1 - 1e-5)
            au, ll = roc_auc_score(t25["whiff"], pm), log_loss(t25["whiff"], pm)
            bmask = (t25["in_play"] == 1) & t25["xwoba"].notna()
            yv, pv = t25.loc[bmask, "xwoba"].to_numpy(), p.loc[bmask, "xwcon"].to_numpy()
            r2 = 1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
            print(f"  {nm:18s} AUC {au:.4f}  logloss {ll:.4f}  타구 질 R² {r2:.4f}")
            acc.append({"model": nm, "auc": au, "ll": ll, "r2_xwcon": r2})
        pd.DataFrame(rows).to_csv("val_swing_hier_summary.csv", index=False)
        pd.DataFrame(grow).to_csv("val_swing_hier_groups.csv", index=False)
        pd.DataFrame(acc).to_csv("val_swing_hier_acc.csv", index=False)
        print("\n저장: val_swing_hier_summary.csv, val_swing_hier_groups.csv, val_swing_hier_acc.csv")
