"""
공격용 모델: 이 공은 치는 게 이득인가, 참는 게 이득인가

  - 스윙 모델: 타자가 배트를 냈을 때의 실점 변화 (헛스윙·파울·타구 결과 포함)
  - 참기 모델: 배트를 안 냈을 때의 실점 변화 (볼 / 루킹 스트라이크)
  - 스윙 가치 = 스윙 모델 - 참기 모델  →  +면 치는 게 이득(노릴 공), -면 참는 게 이득

정답은 build_matchup.py와 같은 '주자·아웃·카운트 반영 실점 변화'(타자 입장에선 +가 좋음)

실행: python build_offense.py   → offense_model.pkl
"""
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

import build_matchup as B

F = B.FEATURES + B.ALIGN_F


def fit(X, y):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63,
                             min_child_samples=300, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1).fit(X, y)


if __name__ == "__main__":
    df = B.load("statcast_2024-04-01_2024-09-30.csv")
    first, second = df[df["game_date"] < "2024-07-01"], df[df["game_date"] >= "2024-07-01"]
    d = B.attach(second, B.batter_profile(first))
    sw, tk = d[d["swing"] == 1], d[d["swing"] == 0]
    print(f"스윙 {len(sw):,}개 / 참은 공 {len(tk):,}개로 학습")
    m_sw = fit(sw[F], sw[B.TARGET])
    m_tk = fit(tk[F], tk[B.TARGET])

    # 상식 점검: 한가운데 vs 낮게 빠지는 공, 카운트별 스윙 가치
    print("\n[점검] 평균적인 포심/슬라이더, 스윙 가치 (100구당, +면 치는 게 이득)")
    base = d.sample(1, random_state=0)[F].iloc[0].to_dict()
    prof1 = B.batter_profile(first)                     # 코스마다 맞는 리그 평균 타자 값으로 채움
    swing_oz, cell_rv = prof1["defaults"]["b_swing_oz"], prof1["cell_league"].to_dict()
    rows = []
    for grp, name, velo, spin, px, pz in [(0, "포심 94", 94, 2300, -7, 15), (1, "슬라이더 85", 85, 2450, 5, 1)]:
        for spot, (x, z) in {"한가운데": (0, 0.5), "낮게 빠짐": (0, -0.25), "바깥 빠짐": (-1.1, 0.5)}.items():
            for b_, s_ in [(0, 0), (3, 1), (0, 2)]:
                r = dict(base, grp=grp, release_speed=velo, release_spin_rate=spin, pfx_x=px, pfx_z=pz,
                         x_in=x, z_norm=z, balls=b_, strikes=s_, outs=0, r1=0, r2=0, r3=0,
                         prev_grp=np.nan, d_velo=np.nan, d_z=np.nan, if_al=0, of_al=0,
                         b_swing_oz=swing_oz[int(B.zone_region(x, z))],
                         b_cell_rv=cell_rv.get(int(B.cell_id(x, z)), 0.0))
                rows.append((name, spot, f"{b_}-{s_}", r))
    X = pd.DataFrame([r for *_, r in rows])[F]
    v = (m_sw.predict(X) - m_tk.predict(X)) * 100
    t = pd.DataFrame({"구종": [a for a, *_ in rows], "코스": [b for _, b, *_ in rows],
                      "카운트": [c for _, _, c, _ in rows], "스윙 가치": v.round(1)})
    print(t.pivot_table(index=["구종", "코스"], columns="카운트", values="스윙 가치", sort=False).to_string())

    with open("offense_model.pkl", "wb") as f:
        pickle.dump({"swing_rv": m_sw, "take_rv": m_tk, "features": F}, f)
    print("\n저장 완료: offense_model.pkl")
