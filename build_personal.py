"""
투수 개인화 2단계

1) 투수 본인의 '릴리스 → 무브먼트' 관계 (personal_move.pkl)
   - 한 시즌 안에서 그 투수의 경기별 평균 릴리스가 달라질 때 그 투수의 무브먼트가 어떻게 같이 움직였나
     (투수·구종 평균과 구장 차이는 빼고)
   - 경기가 적거나 들쭉날쭉한 투수는 리그 평균 쪽으로 당김 (경험적 베이즈 축소, 타자 프로필과 같은 원리)

2) '자기 패스트볼 대비' 피처를 넣은 스윙/헛스윙 모델 (swing_model_v2.pkl, pitch_samples_v2.pkl)
   - 변화구·체인지업의 가치는 자기 주력 패스트볼과 얼마나 다르냐(구속 차, 수직·가로 무브 차)에 크게 달림
   - 이걸 넣으면 '포심을 바꾸면 체인지업까지 같이 좋아지거나 나빠지는' 효과가 투수마다 다르게 계산됨

실행: python build_personal.py move    (1분)
      python build_personal.py model   (10~15분: 검증 + 최종 모델 + 샘플)
"""
import pickle
import sys

import numpy as np
import pandas as pd

import build_matchup as B
import build_release_move as RM

FB_TYPES = ("FF", "SI", "FC")
REL_F = ["fb_dv", "fb_divb", "fb_dhb"]          # 자기 주력 패스트볼 대비 구속·수직무브·가로무브 차
REF_F = ["fb_velo_ref", "fb_ivb_ref", "fb_hb_ref"]


# ------------------------------------------------------------------ 1) 개인 릴리스→무브
def game_level_pitcher(df):
    g = (df.groupby(["season", "pitcher", "pitch_type", "game_pk", "home_team"])
           .agg(n=("velo", "size"), **{c: (c, "mean") for c in RM.X + RM.Y}).reset_index())
    g = g[g["n"] >= 8]
    key = g["season"].astype(str) + "_" + g["pitcher"].astype(str) + "_" + g["pitch_type"]
    park = g["season"].astype(str) + "_" + g["home_team"]
    v = g[RM.X + RM.Y].copy()
    for _ in range(8):
        v = v - v.groupby(key).transform("mean")
        v = v - v.groupby(park).transform("mean")
    v[["pitcher", "pitch_type", "n"]] = g[["pitcher", "pitch_type", "n"]].values
    return v


def personal_coefs(v, prior, min_games=12):
    """투수·구종별 계수 → 리그 계수(prior[(type, y)] = array3) 쪽으로 축소"""
    raw = []
    for (pid, t), s in v.groupby(["pitcher", "pitch_type"]):
        s = s[(s[RM.X].abs() < 1.0).all(axis=1) & (s["ivb"].abs() < 8) & (s["hb"].abs() < 8)]
        if len(s) < min_games:
            continue
        X = s[RM.X].to_numpy(float)
        w = s["n"].to_numpy(float)
        for y in ("ivb", "hb"):
            c, se = RM.wls(X, s[y].to_numpy(float), w)
            raw.append({"pitcher": pid, "pitch_type": t, "y": y, "games": len(s),
                        **{f"b_{x}": c[i] for i, x in enumerate(RM.X)},
                        **{f"se_{x}": se[i] for i, x in enumerate(RM.X)}})
    raw = pd.DataFrame(raw)
    out, info = {}, []
    for (t, y), g in raw.groupby(["pitch_type", "y"]):
        pr = prior.get((t, y))
        if pr is None:
            pr = prior.get((RM.GROUP.get(t, "fb"), y), np.zeros(3))
        for i, x in enumerate(RM.X):
            b, se = g[f"b_{x}"], g[f"se_{x}"].clip(lower=1e-3)
            tau2 = max(np.var(b) - np.mean(se ** 2), 0.05 ** 2)       # 투수 간 진짜 차이의 분산
            wgt = tau2 / (tau2 + se ** 2)
            raw.loc[g.index, f"s_{x}"] = wgt * b + (1 - wgt) * pr[i]
            raw.loc[g.index, f"w_{x}"] = wgt
    for _, r in raw.iterrows():
        out.setdefault((r["pitcher"], r["pitch_type"]), {})[r["y"]] = np.array([r[f"s_{x}"] for x in RM.X])
    return out, raw


def league_prior_from_games(v):
    fit = RM.fit(v.drop(columns=["pitcher"]).assign(w=v["n"].astype(float)), "game")
    return {(r["type"], r["y"]): np.array([r["b_rel_z"], r["b_rel_side"], r["b_ext"]]) for _, r in fit.iterrows()}


# ------------------------------------------------------------------ 2) 자기 패스트볼 대비 피처
def add_fb_ref(df):
    """투수별 주력 패스트볼(FF/SI/FC 중 가장 많이 던진 것)의 평균 구속·무브 → 대비 피처"""
    fb = df[df["pitch_type"].isin(FB_TYPES)]
    cnt = fb.groupby(["pitcher", "pitch_type"]).size().reset_index(name="n")
    main = cnt.sort_values("n").drop_duplicates("pitcher", keep="last").set_index("pitcher")["pitch_type"]
    ref = (fb.merge(main.rename("main_fb"), left_on="pitcher", right_index=True)
             .query("pitch_type == main_fb")
             .groupby("pitcher").agg(fb_velo_ref=("release_speed", "mean"), fb_ivb_ref=("pfx_z", "mean"),
                                     fb_hb_ref=("pfx_x", "mean")))
    df = df.drop(columns=[c for c in REF_F + REL_F + ["main_fb"] if c in df.columns])
    df = df.merge(ref, left_on="pitcher", right_index=True, how="left")
    df["main_fb"] = df["pitcher"].map(main)
    return rel_features(df)


def rel_features(df):
    df["fb_dv"] = df["release_speed"] - df["fb_velo_ref"]
    df["fb_divb"] = df["pfx_z"] - df["fb_ivb_ref"]
    df["fb_dhb"] = df["pfx_x"] - df["fb_hb_ref"]
    return df


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "move"
    if step == "move":
        d24 = RM.read(2024)
        v24 = game_level_pitcher(d24)
        prior24 = league_prior_from_games(v24)
        pm24, raw24 = personal_coefs(v24, prior24)                       # 검증용 (2024만)
        d25 = RM.read(2025)
        v_all = game_level_pitcher(pd.concat([d24, d25], ignore_index=True))
        import mechanics as MC
        prior_app = MC.load_move("release_move.pkl")                     # 앱용: 두 방법 합의 계수가 기준
        pm_all, raw_all = personal_coefs(v_all, prior_app)
        with open("personal_move.pkl", "wb") as f:
            pickle.dump({"app": pm_all, "val2024": pm24, "prior2024": prior24}, f)
        print(f"개인 계수: 검증용 {len(pm24)}개, 앱용 {len(pm_all)}개 (투수×구종)")
        print("개인 데이터 반영 비율(평균 가중치):",
              raw_all.groupby("y")[[f"w_{x}" for x in RM.X]].mean().round(2).to_dict("index"))
        for nm, pid in [("Skenes", 694973), ("Hader", 623352)]:
            for t in ("FF", "SI", "FS", "SL"):
                if (pid, t) in pm_all:
                    print(nm, t, {y: pm_all[(pid, t)][y].round(2).tolist() for y in pm_all[(pid, t)]},
                          "리그", {y: (prior_app.get((t, y)) if prior_app.get((t, y)) is not None else np.zeros(3)).round(2).tolist()
                                 for y in ("ivb", "hb")})
        sys.exit()

    # ---------------- step == "model" (또는 "samples": 샘플만 다시)
    import lightgbm as lgb  # noqa: F401
    from sklearn.metrics import roc_auc_score, log_loss
    import build_swing as W
    import build_mechanics_data as MD

    df = B.load("statcast_2024-04-01_2024-09-30.csv")
    df = add_fb_ref(df)
    F2 = W.FULL_F + REL_F
    if step == "model":
        first, second = df[df["game_date"] < "2024-07-01"], df[df["game_date"] >= "2024-07-01"]
        d = B.attach(second, B.batter_profile(first))
        cut = d["game_date"].max() - pd.Timedelta(days=21)
        tr, te = d[d["game_date"] < cut], d[d["game_date"] >= cut]
        print("[검증] 자기 패스트볼 대비 피처를 넣으면? (마지막 3주 테스트)")
        for target, a, b in [("스윙", tr, te), ("헛스윙", tr[tr["swing"] == 1], te[te["swing"] == 1])]:
            y = "swing" if target == "스윙" else "whiff"
            for name, feats in [("기존", W.FULL_F), ("+패스트볼 대비", F2)]:
                m = W.clf(300).fit(a[feats], a[y])
                p = m.predict_proba(b[feats])[:, 1]
                print(f"  {target:<6}{name:<14} AUC {roc_auc_score(b[y], p):.4f}  logloss {log_loss(b[y], p):.4f}")
            if target == "헛스윙":
                for grp, lab in ((1, "변화구"), (2, "오프스피드")):
                    bb = b[b["grp"] == grp]
                    for name, feats in [("기존", W.FULL_F), ("+패스트볼 대비", F2)]:
                        m = W.clf(300).fit(a[feats], a[y])
                        print(f"     {lab} {name:<14} AUC {roc_auc_score(bb[y], m.predict_proba(bb[feats])[:, 1]):.4f}")
        print("최종 모델 학습 중...")
        swing_m = W.clf().fit(d[F2], d["swing"])
        sw = d[d["swing"] == 1]
        whiff_m = W.clf().fit(sw[F2], sw["whiff"])
        with open("swing_model_v2.pkl", "wb") as f:
            pickle.dump({"swing": swing_m, "whiff": whiff_m, "features": F2, "rel": REL_F}, f)
    # 샘플 (build_mechanics_data와 같은 방식 + 패스트볼 기준값)
    prof = B.batter_profile(df)
    big = df.groupby("pitcher")["pitcher"].transform("size") >= MD.MIN_PITCHER
    dd = df[big]
    dd = dd[dd.groupby(["pitcher", "pitch_type"])["pitch_type"].transform("size") >= MD.MIN_TYPE]
    samp = dd.sample(frac=1, random_state=0).groupby(["pitcher", "pitch_type"]).head(MD.PER_TYPE)
    samp = B.attach(samp, prof)
    keep = ["pitcher", "pitch_type", "stand", "main_fb"] + REF_F + [c for c in F2 if c != "pitcher"]
    samp = samp[list(dict.fromkeys(keep))].copy()
    num = samp.select_dtypes("number").columns.drop("pitcher")
    samp[num] = samp[num].astype(np.float32)
    with open("pitch_samples_v2.pkl", "wb") as f:
        pickle.dump(samp.reset_index(drop=True), f)
    print("저장: swing_model_v2.pkl, pitch_samples_v2.pkl", len(samp))
