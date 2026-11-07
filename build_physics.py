"""
투수·구종별 평균 물리값 저장 (궤적 그리기용)

실행: python build_physics.py   → physics.pkl
"""
import pickle

import pandas as pd

import trajectory as T

COLS = ["pitcher", "player_name", "pitch_type", "p_throws", "release_pos_x", "release_pos_y",
        "release_pos_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "release_speed", "arm_angle"]

if __name__ == "__main__":
    df = pd.read_csv("statcast_2024-04-01_2024-09-30.csv", usecols=COLS)
    df = df.dropna(subset=[c for c in COLS if c != "arm_angle"])     # 팔 각도는 일부 비어 있어도 됨
    phys = (df.groupby(["pitcher", "player_name", "pitch_type", "p_throws"])
              .agg(n=("vx0", "size"), **{c: (c, "mean") for c in COLS[4:]})
              .reset_index())
    phys = phys[phys["n"] >= 30]

    # 점검: 평균 물리값으로 평균 도착 지점을 겨냥했을 때 궤적이 제대로 도착하나
    p = phys.sort_values("n", ascending=False).iloc[0].to_dict()
    tr = T.path(p, p["plate_x"], p["plate_z"])
    print(f"점검: {p['player_name']} {p['pitch_type']}  도착 {tr[1][-1]:+.2f}, {tr[3][-1]:.2f}  "
          f"(목표 {p['plate_x']:+.2f}, {p['plate_z']:.2f}),  비행시간 {tr[0][-1]:.3f}초")
    with open("physics.pkl", "wb") as f:
        pickle.dump(phys, f)
    print(f"저장 완료: physics.pkl  ({phys['pitcher'].nunique()}명, {len(phys)}개 구종)")
