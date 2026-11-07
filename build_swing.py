"""
스윙 / 헛스윙 예측 모델

run value(실점 변화)는 운이 너무 많이 섞여서 미세한 효과가 안 보였음.
그래서 더 직접적인 두 가지를 예측:
  1) 스윙 모델   : 이 공에 타자가 배트를 낼까?
  2) 헛스윙 모델 : 배트를 냈다면 헛칠까?   (스윙한 공만으로 학습)

그리고 질문 두 개에 답함:
  A) 직전 공 정보를 넣으면 예측이 좋아지나?  (AUC 비교)
  B) '높은 직구 → 낮은 변화구'가 정말 효과 있나?
     같은 공을 놓고 '직전 공'만 바꿔서 모델에게 물어봄 (반사실 실험)

실행:
  python build_swing.py               # 검증 + 최종 모델 저장 (swing_model.pkl)
  python build_swing.py --only check  # 검증만
  python build_swing.py --only save   # 모델 저장만 (빠름)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

import build_matchup as B

PREV_ALL = ["prev_grp", "prev_velo", "prev_x", "prev_z",
            "d_velo", "d_x", "d_z", "d_pfx_x", "d_pfx_z"]
BASE_F = B.PITCH_F + B.RELEASE_F + B.BATTER_F + B.SPRAY_F + B.SIT_F
FULL_F = BASE_F + PREV_ALL


def clf(n=400):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=63,
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1)


def check(d):
    """A) 마지막 3주로 테스트: 직전 공 정보가 도움이 되나"""
    cut = d["game_date"].max() - pd.Timedelta(days=21)
    tr, te = d[d["game_date"] < cut], d[d["game_date"] >= cut]
    print("\n[검증 A] 직전 공 정보를 넣으면 좋아지나? (마지막 3주 테스트)")
    print(f"  {'모델':<12}{'피처':<14}{'AUC':>7}{'logloss':>9}")
    models = {}
    for target, sub_tr, sub_te in [("스윙", tr, te),
                                   ("헛스윙", tr[tr["swing"] == 1], te[te["swing"] == 1])]:
        y = "swing" if target == "스윙" else "whiff"
        for name, feats in [("직전 공 없이", BASE_F), ("직전 공 포함", FULL_F)]:
            m = clf(300).fit(sub_tr[feats], sub_tr[y])
            p = m.predict_proba(sub_te[feats])[:, 1]
            print(f"  {target:<12}{name:<14}{roc_auc_score(sub_te[y], p):>7.4f}"
                  f"{log_loss(sub_te[y], p):>9.4f}")
            models[(target, name)] = m
    print("  (AUC는 높을수록, logloss는 낮을수록 좋음)")
    return models[("헛스윙", "직전 공 포함")], te


def counterfactual(model, te, fb):
    """B) 테스트셋의 '낮은 변화구'들에 대해 직전 공만 바꿔 넣고 헛스윙 확률 비교"""
    low = te[(te["grp"] == 1) & (te["z_norm"] < 0.33) & (te["swing"] == 1)].copy()
    low = low.join(fb, on="pitcher").dropna(subset=["fb_velo"])

    def scenario(prev_grp, prev_z, use_fb):
        s = low.copy()
        s["prev_grp"] = prev_grp
        s["prev_velo"] = s["fb_velo"] if use_fb else s["release_speed"]
        pfx_x = s["fb_pfx_x"] if use_fb else s["pfx_x"]
        pfx_z = s["fb_pfx_z"] if use_fb else s["pfx_z"]
        s["prev_x"], s["prev_z"] = s["x_in"], prev_z
        s["d_velo"] = s["release_speed"] - s["prev_velo"]
        s["d_x"], s["d_z"] = 0.0, s["z_norm"] - prev_z
        s["d_pfx_x"], s["d_pfx_z"] = s["pfx_x"] - pfx_x, s["pfx_z"] - pfx_z
        return model.predict_proba(s[FULL_F])[:, 1].mean()

    rows = {"높은 패스트볼 다음": scenario(0, 0.85, True),
            "낮은 패스트볼 다음": scenario(0, 0.15, True),
            "똑같은 낮은 변화구 다음": scenario(1, 0.15, False),
            "높은 변화구 다음": scenario(1, 0.85, False)}
    for c in PREV_ALL:
        low[c] = np.nan
    rows["직전 공 모름"] = model.predict_proba(low[FULL_F])[:, 1].mean()

    print(f"\n[검증 B] 같은 '낮은 변화구' {len(low):,}개에 직전 공만 바꿔 넣었을 때 헛스윙 확률")
    for k, v in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="statcast_2024-04-01_2024-09-30.csv")
    ap.add_argument("--split", default="2024-07-01")
    ap.add_argument("--only", choices=["check", "save"])
    a = ap.parse_args()

    df = B.load(a.data)
    first, second = df[df["game_date"] < a.split], df[df["game_date"] >= a.split]
    d = B.attach(second, B.batter_profile(first))

    # 투수별 평균 패스트볼 (반사실 실험용)
    fb = (df[df["grp"] == 0].groupby("pitcher")
            .agg(fb_velo=("release_speed", "mean"), fb_pfx_x=("pfx_x", "mean"),
                 fb_pfx_z=("pfx_z", "mean")))

    if a.only != "save":
        whiff_m, te = check(d)
        counterfactual(whiff_m, te, fb)
    if a.only != "check":
        print("최종 모델 학습 중...")
        swing_m = clf().fit(d[FULL_F], d["swing"])
        sw = d[d["swing"] == 1]
        whiff_m = clf().fit(sw[FULL_F], sw["whiff"])
        with open("swing_model.pkl", "wb") as f:
            pickle.dump({"swing": swing_m, "whiff": whiff_m, "features": FULL_F}, f)
        print("저장 완료: swing_model.pkl")
