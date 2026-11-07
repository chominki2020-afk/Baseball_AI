"""
타구 가치 모델: 이 타구(속도·각도·방향)가 이 수비 배치에서 얼마짜리인가

  - 입력: 타구속도, 발사각, 타구 방향(당긴 쪽 +), 좌/우타, 내야·외야 수비 배치
  - 정답: 그 타구의 실제 결과 가치 (wOBA 값: 아웃 0, 단타 약 0.9, 홈런 약 2.0)
  - 같은 타구라도 수비가 어디 서 있느냐에 따라 안타/아웃이 갈림 → 수비 효과를 직접 학습

쓰임
  - 수비팀: 이 타자의 실제 타구들을 각 수비 배치에 넣어 보고, 가장 적게 맞는 배치를 추천
  - 공격팀: 상대 수비 배치에서, 이 타자 타구를 당겨칠 때 / 밀어칠 때 가치 비교

실행: python build_bip.py   → bip_model.pkl
"""
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

import build_matchup as B

F = ["launch_speed", "launch_angle", "spray_pull", "lefty", "if_al", "of_al"]
WOBA_SCALE = 1.25          # wOBA 차이 → 실점 차이 환산 (대략 1.2~1.3)


def prep(df):
    b = df[(df["description"] == "hit_into_play")].dropna(
        subset=["launch_speed", "launch_angle", "spray_pull", "woba_value"]).copy()
    b["lefty"] = (b["stand"] == "L").astype(int)
    return b


def fit(b):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                          min_child_samples=80, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.9, verbose=-1)
    return m.fit(b[F], b["woba_value"])


def check(b):
    cut = b["game_date"].max() - pd.Timedelta(days=30)
    tr, te = b[b["game_date"] < cut], b[b["game_date"] >= cut]
    base = ((te["woba_value"] - tr["woba_value"].mean()) ** 2).mean()
    print("\n[검증] 마지막 30일 타구로 테스트 (설명력 R²)")
    for name, feats in [("속도·각도만 (= 기대 타율 방식)", ["launch_speed", "launch_angle"]),
                        ("+ 타구 방향", ["launch_speed", "launch_angle", "spray_pull", "lefty"]),
                        ("+ 수비 배치", F)]:
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=63,
                              min_child_samples=80, verbose=-1).fit(tr[feats], tr["woba_value"])
        r2 = 1 - ((te["woba_value"] - m.predict(te[feats])) ** 2).mean() / base
        print(f"  {name:<24} {r2:.4f}")


def show_effect(m):
    """대표 타구로 수비 효과 확인: 좌타자의 당겨친 강한 땅볼"""
    print("\n[수비 효과] 같은 타구, 수비 배치만 바꿨을 때 예상 wOBA")
    cases = [("좌타 · 당겨친 강한 땅볼 (100마일, 각도 0°, 방향 +20°)", 100, 0, 20, 1),
             ("좌타 · 밀어친 강한 땅볼 (100마일, 각도 0°, 방향 -20°)", 100, 0, -20, 1),
             ("우타 · 당겨친 강한 땅볼 (100마일, 각도 0°, 방향 +20°)", 100, 0, 20, 0)]
    for label, ev, la, sp, l in cases:
        vals = []
        for i in range(3):
            x = pd.DataFrame([[ev, la, sp, l, i, 0]], columns=F)
            vals.append(f"{B.IF_KO[i]} {m.predict(x)[0]:.3f}")
        print(f"  {label}\n     " + "  |  ".join(vals))


if __name__ == "__main__":
    df = B.load("statcast_2024-04-01_2024-09-30.csv")
    b = prep(df)
    print(f"타구 {len(b):,}개")
    check(b)
    m = fit(b)
    show_effect(m)
    keep = b[["batter", "stand", "launch_speed", "launch_angle", "spray_pull", "lefty"]].reset_index(drop=True)
    with open("bip_model.pkl", "wb") as f:
        pickle.dump({"model": m, "features": F, "batter_bip": keep,
                     "lg_woba": b["woba_value"].mean(), "scale": WOBA_SCALE}, f)
    print("\n저장 완료: bip_model.pkl")
