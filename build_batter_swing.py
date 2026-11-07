"""
타자 스윙 목표 — 배트 트래킹(스윙마다 측정) 값으로 '스윙을 이렇게 바꾸면 결과가 어떻게 바뀌나'

스윙 값 (스탯캐스트 배트 트래킹, 2024~)
  bat_speed        배트 스피드 (mph)
  swing_length     스윙 길이 (피트, 짧을수록 콤팩트)
  attack_angle     어택 앵글 (°, +면 올려 치기)
  attack_direction 어택 디렉션 (°, -면 당기는 방향)
  swing_path_tilt  스윙 경로 기울기 (°, 클수록 가파름)
  contact_y        컨택 포인트 (인치, 클수록 몸 앞에서 맞힘)

결과 = 세 단계 모델
  헛스윙 확률 | 스윙  →  (맞혔을 때) 인플레이 확률  →  인플레이 xwOBA (타구 속도·각도 기준 기대 가치)
  스윙당 xwOBA = (1 - 헛스윙) × 인플레이 × xwOBAcon

두 가지 방식으로 학습해서 2024 → 2025 실제로 스윙을 바꾼 타자들로 검증 (투수 개선 때와 같은 자연 실험)
  A '스윙마다' : 그 스윙의 실제 측정값을 피처로 (스윙 간 차이 — 타이밍·공 높이 적응이 섞임)
  B '타자 스타일': 그 타자의 시즌 평균 스윙 값을 피처로 (타자 간 차이 — 타자 실력이 섞임)

실행: python build_batter_swing.py val    (학습 + 검증, 5~10분)
      python build_batter_swing.py save   (앱용 최종 모델·리포트·샘플 저장)
"""
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import build_matchup as B

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
RAW = {"bat_speed": "bat_speed", "swing_length": "swing_length", "attack_angle": "attack_angle",
       "attack_direction": "attack_direction", "swing_path_tilt": "swing_path_tilt",
       "intercept_ball_minus_batter_pos_y_inches": "contact_y"}
SW = list(RAW.values())
PITCH_F = ["grp", "release_speed", "pfx_x", "pfx_z", "x_in", "z_norm", "balls", "strikes", "same_hand", "vaa"]
STYLE = ["m_" + c for c in SW]                 # 타자 시즌 평균
MIN_SW = 300                                   # 검증: 두 시즌 모두 이 이상 스윙한 타자


def load(year):
    for c in list(RAW) + ["estimated_woba_using_speedangle"]:
        if c not in B.COLS:
            B.COLS.append(c)
    df = B.load(FILES[year])
    df = df.rename(columns=RAW)
    sw = df[(df["swing"] == 1) & df["bat_speed"].notna() & (df["bat_speed"] >= 50)
            & ~df["description"].isin(["foul_bunt", "missed_bunt"])].copy()
    sw["contact"] = 1 - sw["whiff"]
    sw["xwoba"] = sw["estimated_woba_using_speedangle"]
    sw["season"] = year
    for c in SW:
        sw["m_" + c] = sw.groupby("batter")[c].transform("mean")
    return sw


def portable(BS):
    """다른 컴퓨터(다른 pandas·sklearn 버전)에서도 열리게: 표는 numpy 배열로, 선형 부분은 숫자 계수로"""
    out = dict(BS)
    for k in ("report", "samples"):
        df = BS[k]
        out[k] = {"__df__": True, "index": df.index.to_numpy(), "index_name": df.index.name,
                  "cols": {c: df[c].to_numpy() for c in df.columns}}
    m = dict(BS["model"])
    for k in ("miss", "bip", "xwcon"):
        p = m[k]
        if not isinstance(p, dict):
            m[k] = {"coef": np.asarray(p.coef_, float).ravel(), "b": float(np.ravel(p.intercept_)[0])}
    m["mu"], m["sd"] = np.asarray(m["mu"], float), np.asarray(m["sd"], float)
    out["model"] = m
    out["bounds"] = {c: (float(a), float(b)) for c, (a, b) in BS["bounds"].items()}
    out["cal"] = {c: float(v) for c, v in BS["cal"].items()}
    out["k_len"] = float(BS["k_len"])
    for k in ("k_aa_cy", "k_dir_cy"):
        if k in BS:
            out[k] = float(BS[k])
    return out


def unpack(BS):
    out = dict(BS)
    for k in ("report", "samples"):
        v = BS[k]
        if isinstance(v, dict) and v.get("__df__"):
            df = pd.DataFrame(v["cols"])
            if k == "report":
                df.index = pd.Index(v["index"], name=v["index_name"])
            out[k] = df
    return out


def reg(n=400):
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def clf(n=400):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit(sw, feats, big_leaf=None):
    """big_leaf: 잎 하나에 최소 이만큼 스윙 (타자 수백 명 중 한 명만 따로 떼어내 외우지 못하게)"""
    m = {"features": feats}
    def c_():
        return clf() if big_leaf is None else lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=big_leaf, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    def r_(n):
        return reg() if big_leaf is None else lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=n, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m["miss"] = c_().fit(sw[feats], sw["whiff"])
    con = sw[sw["contact"] == 1]
    m["bip"] = c_().fit(con[feats], con["in_play"])
    b = con[(con["in_play"] == 1) & con["xwoba"].notna()]
    m["xwcon"] = r_(None if big_leaf is None else big_leaf // 3).fit(b[feats], b["xwoba"])
    return m


# ---- D: 매끄러운 스타일 모델 (공 상황은 GBM, 타자 스타일은 2차식 + 코스와의 상호작용)
#   B(GBM에 시즌 평균을 그냥 넣기)는 타자마다 평균값이 고유해서 '이 타자 = 이 점'을 외워 버림
#   → 지금 값이 항상 최적처럼 보이는 문제. D는 스타일 효과를 몇십 개 계수의 매끄러운 식으로만 표현
PAIRS = [(i, j) for i in range(len(SW)) for j in range(i + 1, len(SW))]


def style_terms(X, mu, sd, inter=True):
    """inter: 스윙 요소끼리의 조합 (예: 배트 스피드 × 어택 앵글 — 느린 배트로 올려 치면 약한 뜬공)"""
    z = (X[STYLE].to_numpy(float) - mu) / sd
    zn = X["z_norm"].to_numpy(float)[:, None] - 0.5
    xi = X["x_in"].to_numpy(float)[:, None]
    parts = [z, z ** 2, z * zn, z * xi]
    if inter:
        parts.append(np.stack([z[:, i] * z[:, j] for i, j in PAIRS], axis=1))
    return np.hstack(parts)


def fit_smooth(sw, inter=True):
    from sklearn.linear_model import LogisticRegression, Ridge
    per = sw.groupby("batter")[STYLE].first()
    mu, sd = per.mean().to_numpy(), per.std().to_numpy()
    m = {"kind": "smooth", "mu": mu, "sd": sd, "features": PITCH_F + STYLE, "inter": inter}
    st_ = lambda X: style_terms(X, mu, sd, inter)
    eps = 1e-4
    def lg(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))[:, None]
    m["miss0"] = clf().fit(sw[PITCH_F], sw["whiff"])
    m["miss"] = LogisticRegression(C=1.0, max_iter=500).fit(
        np.hstack([lg(m["miss0"].predict_proba(sw[PITCH_F])[:, 1]), st_(sw)]), sw["whiff"])
    con = sw[sw["contact"] == 1]
    m["bip0"] = clf().fit(con[PITCH_F], con["in_play"])
    m["bip"] = LogisticRegression(C=1.0, max_iter=500).fit(
        np.hstack([lg(m["bip0"].predict_proba(con[PITCH_F])[:, 1]), st_(con)]), con["in_play"])
    b = con[(con["in_play"] == 1) & con["xwoba"].notna()]
    m["xw0"] = reg().fit(b[PITCH_F], b["xwoba"])
    m["xwcon"] = Ridge(alpha=1.0).fit(np.hstack([m["xw0"].predict(b[PITCH_F])[:, None], st_(b)]),
                                      b["xwoba"])
    return m


def _lin(part, Z):
    """sklearn 모델 또는 {'coef','b'} 숫자 계수 (다른 컴퓨터의 sklearn 버전과 상관없이 쓰려고 숫자로 저장)"""
    if isinstance(part, dict):
        return Z @ part["coef"] + part["b"]
    if hasattr(part, "predict_proba"):
        return part.decision_function(Z)
    return part.predict(Z)


def predict_smooth(m, X):
    eps = 1e-4
    def lg(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))[:, None]
    sig = lambda z: 1 / (1 + np.exp(-z))
    T = style_terms(X, m["mu"], m["sd"], m.get("inter", False))
    miss = sig(_lin(m["miss"], np.hstack([lg(m["miss0"].predict_proba(X[PITCH_F])[:, 1]), T])))
    bip = sig(_lin(m["bip"], np.hstack([lg(m["bip0"].predict_proba(X[PITCH_F])[:, 1]), T])))
    xw = _lin(m["xwcon"], np.hstack([m["xw0"].predict(X[PITCH_F])[:, None], T]))
    return pd.DataFrame({"miss": miss, "bip": bip, "xwcon": xw, "xw_sw": (1 - miss) * bip * xw}, index=X.index)


def predict(m, X):
    if m.get("kind") == "smooth":
        return predict_smooth(m, X)
    X = X[m["features"]]
    miss = m["miss"].predict_proba(X)[:, 1]
    bip = m["bip"].predict_proba(X)[:, 1]
    xw = m["xwcon"].predict(X)
    return pd.DataFrame({"miss": miss, "bip": bip, "xwcon": xw, "xw_sw": (1 - miss) * bip * xw}, index=X.index)


def batter_summary(sw):
    g = sw.groupby("batter")
    t = g.agg(n=("bat_speed", "size"), **{c: (c, "mean") for c in SW}, whiff=("whiff", "mean"),
              bip=("in_play", "sum"), xw_sum=("xwoba", "sum"), stand=("stand", lambda s: s.mode().iloc[0]))
    con = sw[sw["contact"] == 1].groupby("batter")["in_play"].mean()
    t["bip_rate"] = con.reindex(t.index)
    t["xwcon"] = t["xw_sum"] / t["bip"].clip(lower=1)
    t["xw_sw"] = t["xw_sum"] / t["n"]
    return t


def corr_ci(x, y, B_=2000, seed=0):
    rng = np.random.default_rng(seed)
    r = np.corrcoef(x, y)[0, 1]
    bs = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, len(x), len(x)) for _ in range(B_))]
    return r, np.percentile(bs, [2.5, 97.5])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "val"
    s24 = load(2024)
    if step == "val":
        s25 = load(2025)
        print(f"스윙: 2024 {len(s24):,} / 2025 {len(s25):,}")
        A = fit(s24, PITCH_F + SW)
        Bm = fit(s24, PITCH_F + STYLE)
        Dm = fit_smooth(s24, inter=False)
        Fm = fit_smooth(s24, inter=True)
        Em = fit(s24, PITCH_F + STYLE, big_leaf=6000)
        a, b = batter_summary(s24), batter_summary(s25)
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["n_a"] >= MIN_SW) & (j["n_b"] >= MIN_SW)]
        for c in SW + ["whiff", "xwcon", "xw_sw", "bip_rate"]:
            j["d_" + c] = j[c + "_b"] - j[c + "_a"]
            j["d_" + c] -= j["d_" + c].median()                 # 리그 전체 해마다 변화 제거
        print(f"두 시즌 모두 {MIN_SW}스윙 이상 타자: {len(j)}명")
        print("스윙 변화 크기(중앙값 절대값):", {c: round(j["d_" + c].abs().median(), 2) for c in SW})
        # 예측: 2024 스윙들에 실제 변화량만 더함
        smp = s24[s24["batter"].isin(j.index)].groupby("batter", group_keys=False).sample(frac=1, random_state=0)
        smp = smp.groupby("batter").head(250)
        shift = j[["d_" + c for c in SW]].rename(columns=lambda c: c[2:])
        base_A, base_B, base_D, base_E = predict(A, smp), predict(Bm, smp), predict(Dm, smp), predict(Em, smp)
        base_F = predict(Fm, smp)
        x = smp.copy()
        for c in SW:
            add = x["batter"].map(shift[c])
            x[c] = x[c] + add
            x["m_" + c] = x["m_" + c] + add
        new_A, new_B, new_D, new_E = predict(A, x), predict(Bm, x), predict(Dm, x), predict(Em, x)
        new_F = predict(Fm, x)
        out = {}
        for nm, b0, b1 in (("A 스윙마다", base_A, new_A), ("B 타자 스타일", base_B, new_B),
                           ("D 매끄러운 스타일", base_D, new_D), ("E 큰 잎 GBM", base_E, new_E),
                           ("F 스타일+조합", base_F, new_F)):
            dlt = (b1 - b0).groupby(smp["batter"]).mean()
            out[nm] = dlt - dlt.median()
        # 측정 오차 (실제값의 표본 오차) — 스윙당 xwOBA는 부트스트랩 대신 근사
        se = {"whiff": np.sqrt(j["whiff_a"] * (1 - j["whiff_a"]) / j["n_a"] + j["whiff_b"] * (1 - j["whiff_b"]) / j["n_b"])}
        rows = []
        for tgt, lab, col in (("whiff", "헛스윙%(스윙당)", "miss"), ("xwcon", "인플레이 xwOBA", "xwcon"),
                              ("xw_sw", "스윙당 xwOBA", "xw_sw"), ("bip_rate", "맞혔을 때 인플레이%", "bip")):
            act = j["d_" + tgt].to_numpy()
            print(f"\n=== {lab}")
            if tgt == "whiff":
                v_true = act.var() - (se["whiff"] ** 2).mean()
                print(f"   (측정 오차 때문에 최대 가능 상관 ≈{np.sqrt(max(v_true, 0) / act.var()):.2f})")
            for nm, d in out.items():
                p = d[col].reindex(j.index).to_numpy()
                r, ci = corr_ci(p, act)
                slope = np.polyfit(p, act, 1)[0]
                print(f"  {nm:12s} 상관 {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  기울기 {slope:+.2f}")
                rows.append({"target": tgt, "label": lab, "method": nm, "n": len(j), "r": r, "lo": ci[0], "hi": ci[1],
                             "slope": slope})
        # 참고: 모델 없이 '스윙 값 변화'만으로 단순 회귀했을 때 (2024 타자 간 관계로 학습 → 변화에 적용)
        from numpy.linalg import lstsq
        Xa = np.c_[np.ones(len(a)), a[SW].to_numpy()]
        for tgt, lab in (("whiff", "헛스윙%"), ("xwcon", "인플레이 xwOBA"), ("xw_sw", "스윙당 xwOBA")):
            ok = a["n"] >= MIN_SW
            coef = lstsq(Xa[ok.to_numpy()], a.loc[ok, tgt].to_numpy(), rcond=None)[0]
            p = j[["d_" + c for c in SW]].to_numpy() @ coef[1:]
            r, ci = corr_ci(p, j["d_" + tgt].to_numpy())
            print(f"  [참고] 타자 간 단순 회귀 → {lab}: 상관 {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
            rows.append({"target": tgt, "label": lab, "method": "C 단순 회귀", "n": len(j), "r": r, "lo": ci[0],
                         "hi": ci[1], "slope": np.nan})
        pd.DataFrame(rows).to_csv("val_batter_summary.csv", index=False)
        j.to_csv("val_batter_result.csv")
        with open("batter_val_models.pkl", "wb") as f:
            pickle.dump({"A": A, "B": Bm, "D": Dm, "E": Em, "F": Fm}, f)
        # ---- 타자 유형별: 어택 앵글 1° 올리면? (실제 vs 모델)
        j["grp"] = pd.qcut(j["bat_speed_a"], 3, labels=["컨택형(느린 배트)", "중간", "파워형(빠른 배트)"])
        sm2 = smp.copy()
        sm2["m_attack_angle"] = sm2["m_attack_angle"] + 1
        eff = {nm: (predict(m_, sm2) - predict(m_, smp)).groupby(smp["batter"]).mean()
               for nm, m_ in (("D", Dm), ("F", Fm))}
        print("\n[타자 유형별] 어택 앵글 1° 올리면 — 실제(시즌 간 변화 회귀) vs 모델")
        grows = []
        rng = np.random.default_rng(0)
        for g, d in j.groupby("grp", observed=True):
            line = f"  {g:12s}"
            for tgt, col, sc, u in (("d_whiff", "miss", 100, "%p"), ("d_xwcon", "xwcon", 1000, ""), ("d_xw_sw", "xw_sw", 1000, "")):
                xx, yy = d["d_attack_angle"].to_numpy(), d[tgt].to_numpy()
                b = np.polyfit(xx, yy, 1)[0] * sc
                bs = [np.polyfit(xx[i], yy[i], 1)[0] * sc for i in (rng.integers(0, len(xx), len(xx)) for _ in range(1000))]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                pD = eff["D"][col].reindex(d.index).mean() * sc
                pF = eff["F"][col].reindex(d.index).mean() * sc
                line += f" | {col} 실제 {b:+.2f}[{lo:+.1f},{hi:+.1f}] D {pD:+.2f} F {pF:+.2f}"
                grows.append({"group": str(g), "target": col, "actual": b, "lo": lo, "hi": hi, "D": pD, "F": pF,
                              "n": len(d)})
            print(line)
        pd.DataFrame(grows).to_csv("val_batter_groups.csv", index=False)
        print("\n저장: val_batter_summary.csv")
        # 배트 스피드를 올리면 스윙 길이도 같이 늘어나는 관계 (같은 타자의 시즌 간 변화)
        k = np.polyfit(j["d_bat_speed"], j["d_swing_length"], 1)[0]
        print(f"배트 스피드 1mph ↑ → 스윙 길이 {k:+.3f}피트 (타자들의 실제 시즌 간 변화)")
        cal = pd.DataFrame(rows)
        cal = cal[cal["method"] == "F 스타일+조합"].set_index("target")["slope"].to_dict()
        # 컨택 포인트를 앞으로 옮기면 어택 앵글이 올라가고(배트가 스윙 바닥을 지나 올라가는 중) 방향은 당기는 쪽으로
        k_aa = np.polyfit(j["d_contact_y"], j["d_attack_angle"], 1)[0]
        k_dir = np.polyfit(j["d_contact_y"], j["d_attack_direction"], 1)[0]
        print(f"컨택 포인트 1in 앞 → 어택 앵글 {k_aa:+.2f}°, 스윙 방향 {k_dir:+.2f}° (타자들의 실제 시즌 간 변화)")
        with open("batter_val_extra.pkl", "wb") as f:
            pickle.dump({"k_len": k, "cal": cal, "k_aa_cy": k_aa, "k_dir_cy": k_dir}, f)

    if step == "save":
        s25 = load(2025)
        both = pd.concat([s24, s25], ignore_index=True)
        M = fit_smooth(both, inter=True)                        # 검증한 F 방식, 두 시즌으로 다시 학습
        ex = pickle.load(open("batter_val_extra.pkl", "rb"))
        rep = batter_summary(s25)
        rep = rep[rep["n"] >= 200]
        ok = rep["n"] >= 200
        for c in SW + ["whiff", "xwcon", "xw_sw"]:
            rep[c + "_pct"] = rep.loc[ok, c].rank(pct=True) * 100
        rep["whiff_pct"] = 100 - rep["whiff_pct"] + 100 / len(rep)            # 헛스윙은 낮을수록 좋게
        bounds = {c: (rep[c].quantile(0.025), rep[c].quantile(0.975)) for c in SW}
        smp = s25[s25["batter"].isin(rep.index)].sample(frac=1, random_state=0).groupby("batter").head(150)
        keep = ["batter", "stand", "pitch_type", "description"] + PITCH_F + SW + STYLE
        smp = smp[keep].reset_index(drop=True)
        num = smp.select_dtypes("number").columns.drop("batter")
        smp[num] = smp[num].astype(np.float32)
        with open("batter_swing.pkl", "wb") as f:
            pickle.dump(portable({"model": M, "report": rep, "samples": smp, "bounds": bounds, "k_len": ex["k_len"],
                                  "cal": ex["cal"], "features": PITCH_F + STYLE, "sw": SW,
                                  "k_aa_cy": ex.get("k_aa_cy", 0.0), "k_dir_cy": ex.get("k_dir_cy", 0.0)}), f)
        print(f"저장: batter_swing.pkl  타자 {len(rep)}명, 샘플 {len(smp):,}스윙")
