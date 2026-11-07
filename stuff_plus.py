"""
미니 Stuff+ 만들기 (스탯캐스트 + LightGBM + SHAP)

목표: 공의 '물리적 특성'만 보고 헛스윙 확률을 예측한 뒤,
      투수-구종별 평균을 리그 평균 100 기준으로 환산 → Stuff+

실행:
  pip install pybaseball lightgbm shap matplotlib pandas scikit-learn
  python stuff_plus.py                    # 실제 MLB 데이터 (기본 2024-07-01 ~ 07-14)
  python stuff_plus.py --start 2024-04-01 --end 2024-09-30   # 기간 변경
  python stuff_plus.py --demo             # 인터넷 없이 가짜 데이터로 흐름만 확인
"""
import argparse
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# 헛스윙 판정에 쓰는 결과값
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play", "foul_bunt", "missed_bunt"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

FEATURES = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z",
            "release_extension", "release_pos_x", "release_pos_z",
            "spin_axis", "velo_diff", "ivb_diff", "hb_diff"]


# ---------------------------------------------------------------- 1. 데이터
def load_statcast(start, end):
    # 기간별로 캐시 파일을 따로 저장 → 기간을 바꾸면 자동으로 새로 받음
    cache = f"statcast_{start}_{end}.csv"
    if os.path.exists(cache):
        print(f"캐시 사용: {cache}")
        return pd.read_csv(cache)
    from pybaseball import statcast, cache as pb_cache
    pb_cache.enable()
    print(f"스탯캐스트 다운로드 중: {start} ~ {end} (기간이 길면 몇 분 걸려요)")
    df = statcast(start_dt=start, end_dt=end)
    df.to_csv(cache, index=False)
    return df


def make_demo(n=60000, seed=0):
    """인터넷 없이 테스트용: 그럴듯한 관계를 넣은 가짜 투구 데이터"""
    rng = np.random.default_rng(seed)
    pitchers = [f"Pitcher_{i:02d}" for i in range(80)]
    types = {"FF": (94, 2300, -6, 16), "SL": (85, 2450, 5, 1),
             "CH": (85, 1750, -14, 6), "CU": (79, 2600, 8, -10)}
    rows = []
    for p in pitchers:
        hand = rng.choice(["R", "L"])
        skill = rng.normal(0, 1)
        for t, (v, s, hx, hz) in types.items():
            k = n // (len(pitchers) * len(types))
            velo = rng.normal(v + skill, 1.2, k)
            spin = rng.normal(s + 100 * skill, 120, k)
            px = rng.normal(hx, 2.5, k) / 12
            pz = rng.normal(hz + skill, 2.5, k) / 12
            # 구속·회전·무브먼트가 좋을수록 헛스윙 확률 ↑
            logit = (-1.8 + 0.08 * (velo - v) + 0.002 * (spin - s)
                     + 0.25 * abs(px * 12 - hx) / 3 + 0.3 * skill
                     + (0.4 if t in ("SL", "CH") else 0))
            whiff = rng.random(k) < 1 / (1 + np.exp(-logit))
            swing = rng.random(k) < 0.47
            desc = np.where(~swing, "ball",
                            np.where(whiff, "swinging_strike", "foul"))
            rows.append(pd.DataFrame({
                "player_name": p, "pitcher": hash(p) % 10**6, "p_throws": hand,
                "pitch_type": t, "release_speed": velo, "release_spin_rate": spin,
                "pfx_x": px * (-1 if hand == "L" else 1), "pfx_z": pz,
                "release_extension": rng.normal(6.3, 0.3, k),
                "release_pos_x": rng.normal(-1.8 if hand == "R" else 1.8, 0.3, k),
                "release_pos_z": rng.normal(5.8, 0.3, k),
                "spin_axis": rng.uniform(0, 360, k), "description": desc}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 2. 전처리
def prepare(df):
    cols = ["player_name", "pitcher", "p_throws", "pitch_type", "description",
            "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
            "release_extension", "release_pos_x", "release_pos_z", "spin_axis"]
    df = df[cols].dropna().copy()
    df = df[df["description"].isin(SWING)]           # 스윙한 공만 사용
    df["whiff"] = df["description"].isin(WHIFF).astype(int)

    # 좌투수는 좌우 반전 → 좌/우 투수를 같은 기준으로 비교
    lefty = df["p_throws"] == "L"
    df.loc[lefty, ["pfx_x", "release_pos_x"]] *= -1
    df.loc[lefty, "spin_axis"] = 360 - df.loc[lefty, "spin_axis"]

    # 인치 단위로 변환 (pfx는 피트 단위로 들어옴)
    df["pfx_x"] *= 12
    df["pfx_z"] *= 12

    # 같은 투수의 주 패스트볼 대비 차이 → '피치 터널/구속차' 효과
    fb = (df[df["pitch_type"].isin(["FF", "SI", "FC"])]
          .groupby("pitcher")[["release_speed", "pfx_z", "pfx_x"]].mean()
          .rename(columns={"release_speed": "fb_v", "pfx_z": "fb_z", "pfx_x": "fb_x"}))
    df = df.join(fb, on="pitcher").dropna(subset=["fb_v"])
    df["velo_diff"] = df["release_speed"] - df["fb_v"]
    df["ivb_diff"] = df["pfx_z"] - df["fb_z"]
    df["hb_diff"] = df["pfx_x"] - df["fb_x"]
    return df


# ---------------------------------------------------------------- 3. 모델
# 구종을 3그룹으로 나눠 모델을 따로 학습
#  → "변화구냐 아니냐"가 아니라 "같은 계열 안에서 뭐가 헛스윙을 만드나"를 배우게 됨
GROUPS = {
    "fastball": ["FF", "SI", "FC"],
    "breaking": ["SL", "ST", "CU", "KC", "SV", "CS"],
    "offspeed": ["CH", "FS", "FO", "SC"],
}
GROUP_KO = {"fastball": "패스트볼", "breaking": "변화구", "offspeed": "오프스피드"}


def add_group(df):
    lookup = {t: g for g, ts in GROUPS.items() for t in ts}
    df = df.copy()
    df["group"] = df["pitch_type"].map(lookup)
    return df.dropna(subset=["group"])


def train(df):
    models = {}
    print("\n[모델] 구종 그룹별 검증 AUC  (0.5=찍기, 0.6대면 구위 모델로 무난)")
    for g, part in df.groupby("group"):
        # 같은 투수가 train/test에 동시에 들어가지 않게 투수 단위로 분할
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(split.split(part, groups=part["pitcher"]))
        X, y = part[FEATURES], part["whiff"]
        model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                                   num_leaves=31, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbose=-1)
        model.fit(X.iloc[tr], y.iloc[tr])
        auc = roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1])
        print(f"  {GROUP_KO[g]:6s} ({len(part):>7,}구)  AUC = {auc:.3f}")
        models[g] = model
    return models


def predict(df, models):
    out = pd.Series(index=df.index, dtype=float)
    for g, part in df.groupby("group"):
        out.loc[part.index] = models[g].predict_proba(part[FEATURES])[:, 1]
    return out


# ---------------------------------------------------------------- 4. Stuff+
def stuff_plus(df, models, min_pitches=50):
    df = df.copy()
    df["xwhiff"] = predict(df, models)
    # 구종별 리그 평균 = 100 기준 (슬라이더는 슬라이더끼리, 포심은 포심끼리 비교)
    type_avg = df.groupby("pitch_type")["xwhiff"].transform("mean")
    df["stuff_rel"] = df["xwhiff"] / type_avg * 100

    board = (df.groupby(["player_name", "pitch_type"])
               .agg(pitches=("xwhiff", "size"),
                    xwhiff=("xwhiff", "mean"),
                    actual_whiff=("whiff", "mean"),
                    velo=("release_speed", "mean"),
                    spin=("release_spin_rate", "mean"),
                    stuff_plus=("stuff_rel", "mean"))
               .reset_index())
    board = board[board["pitches"] >= min_pitches]
    board["stuff_plus"] = board["stuff_plus"].round(0).astype(int)
    # 구종 안에서 몇 등인지 (1 = 그 구종 최고)
    board["rank_in_type"] = (board.groupby("pitch_type")["stuff_plus"]
                                  .rank(ascending=False, method="min").astype(int))

    # 투수 종합 Stuff+ : 구종별 점수를 투구 수로 가중평균
    pitcher = (df.groupby("player_name")
                 .agg(pitches=("stuff_rel", "size"), stuff_plus=("stuff_rel", "mean"),
                      arsenal=("pitch_type", lambda s: "/".join(sorted(s.unique()))))
                 .reset_index())
    pitcher = pitcher[pitcher["pitches"] >= min_pitches * 2]
    pitcher["stuff_plus"] = pitcher["stuff_plus"].round(0).astype(int)

    return (board.sort_values("stuff_plus", ascending=False),
            pitcher.sort_values("stuff_plus", ascending=False))


def explain(models, df):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for g, model in models.items():
        part = df[df["group"] == g]
        sample = part[FEATURES].sample(min(3000, len(part)), random_state=0)
        sv = shap.TreeExplainer(model).shap_values(sample)
        sv = sv[1] if isinstance(sv, list) else sv
        plt.figure()
        shap.summary_plot(sv, sample, show=False)
        plt.title(f"{g} model", fontsize=14)
        plt.tight_layout()
        out = f"shap_{g}.png"
        plt.savefig(out, dpi=150)
        plt.close("all")
        print(f"[SHAP] {GROUP_KO[g]} 모델 그림 저장: {out}")


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2024-07-14")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    raw = make_demo() if a.demo else load_statcast(a.start, a.end)
    df = add_group(prepare(raw))
    print(f"스윙한 공 {len(df):,}개, 헛스윙률 {df['whiff'].mean():.1%}")

    models = train(df)
    board, pitcher = stuff_plus(df, models)
    board.to_csv("stuff_plus_leaderboard.csv", index=False, encoding="utf-8-sig")
    pitcher.to_csv("stuff_plus_pitchers.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 140)
    fmt = lambda x: f"{x:.3f}"
    print("\n[구종별 Stuff+ 상위 15]  (100 = 같은 구종 리그 평균)")
    print(board.head(15).to_string(index=False, float_format=fmt))

    print("\n[구종마다 1위]")
    best = board[board["rank_in_type"] == 1].sort_values("pitches", ascending=False)
    print(best[["pitch_type", "player_name", "stuff_plus", "pitches"]].to_string(index=False))

    print("\n[투수 종합 Stuff+ 상위 10]")
    print(pitcher.head(10).to_string(index=False))

    print("\n저장: stuff_plus_leaderboard.csv, stuff_plus_pitchers.csv")
    explain(models, df)
