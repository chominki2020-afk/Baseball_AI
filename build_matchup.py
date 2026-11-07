"""
투수 vs 타자 매치업 모델 만들기  (볼배합 추천기의 '두뇌')

무엇을 하나:
  1) 타자 프로필 : 타자별 존별 강약(핫/콜드존), 구종 계열별 헛스윙률·추격률,
                  타구 품질(xwOBA, 타구속도, 발사각), 당겨치기 비율
  2) 매치업 모델 : 투구 하나가 '실점 기대치(run value)'를 얼마나 바꾸는지 예측
                  입력 = 공의 성질 + 코스 + 카운트 + 타자 프로필
  3) 검증        : 타자 정보를 넣으면 예측이 실제로 좋아지는지 확인

데이터 누수 방지:
  학습용 타자 프로필은 시즌 전반(4~6월)으로 만들고, 모델은 후반(7~9월) 투구로 학습.
  (같은 투구로 프로필도 만들고 학습도 하면 모델이 정답을 미리 보게 됨)

실행:
  python build_matchup.py --data statcast_2024-04-01_2024-09-30.csv
  → matchup_model.pkl 생성 (recommend.py가 이 파일을 씀)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

COLS = ["game_date", "batter", "pitcher", "player_name", "stand", "p_throws",
        "balls", "strikes", "pitch_type", "zone", "plate_x", "plate_z",
        "sz_top", "sz_bot", "description", "launch_speed", "launch_angle",
        "hc_x", "hc_y", "delta_run_exp", "estimated_woba_using_speedangle",
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "game_pk", "at_bat_number", "pitch_number",
        "on_1b", "on_2b", "on_3b", "outs_when_up",
        "inning", "inning_topbot", "bat_score", "post_bat_score",
        "if_fielding_alignment", "of_fielding_alignment", "woba_value",
        "release_pos_x", "release_pos_z", "release_extension", "vy0", "vz0", "ay", "az"]

# 수비 시프트 코드 (2023년부터 극단적 시프트 금지 → 이 세 가지만 존재)
IF_ALIGN = {"Standard": 0, "Infield shade": 1, "Strategic": 2}
OF_ALIGN = {"Standard": 0, "Strategic": 1}
IF_KO = {0: "정상 수비", 1: "내야 쉐이드", 2: "전략적 배치"}
OF_KO = {0: "외야 정상", 1: "외야 전략적 배치"}
ALIGN_F = ["if_al", "of_al"]

# 모델이 맞히는 정답: 주자·아웃·카운트를 반영한 실점 변화 (아래 add_context_rv에서 계산)
TARGET = "rv_ctx"

GROUPS = {"fastball": ["FF", "SI", "FC"],
          "breaking": ["SL", "ST", "CU", "KC", "SV", "CS"],
          "offspeed": ["CH", "FS", "FO", "SC"]}
GROUP_CODE = {"fastball": 0, "breaking": 1, "offspeed": 2}
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play", "foul_bunt", "missed_bunt"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

# 코스 격자 (타자 기준): x_in = 몸쪽(+) / 바깥쪽(-) [피트], z_norm = 존 바닥 0 ~ 존 꼭대기 1
X_EDGES = [-0.83, -0.28, 0.28, 0.83]     # 바깥 빠짐 | 바깥 | 가운데 | 몸쪽 | 몸쪽 빠짐
Z_EDGES = [0.0, 0.33, 0.67, 1.0]         # 낮게 빠짐 | 낮게 | 가운데 | 높게 | 높게 빠짐

# 릴리스 포인트·진입각 (투수의 '던지는 모습' 정보)
#   rel_z 타점 높이, rel_side 몸에서 옆으로(팔 쪽, 피트), ext 익스텐션,
#   vaa_rel 같은 높이로 들어온 같은 계열 공 대비 진입각 차이 (+면 더 평평)
RELEASE_F = ["rel_z", "rel_side", "ext", "vaa_rel"]
# 신체조건 (player_bio.csv, fetch_bio.py로 받음): 키(피트), 몸무게(파운드), 키 대비 타점, 키 대비 익스텐션
BODY_F = ["height", "weight", "rel_z_h", "ext_h"]
ARSENAL_F = ["grp", "release_speed", "release_spin_rate", "pfx_x", "pfx_z"] + RELEASE_F + BODY_F
PITCH_F = ["grp", "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
           "x_in", "z_norm", "balls", "strikes", "same_hand"]
# ※ RELEASE_F는 실점 모델에선 성능이 오히려 떨어져서(R² 0.0300→0.0289, 두 기간 모두) 빼고,
#   헛스윙 예측이 좋아지는(AUC 0.7904→0.7930) 스윙/헛스윙 모델(build_swing.py)에만 넣음
BATTER_F = ["b_cell_rv", "b_whiff", "b_chase", "b_xwoba", "b_ev", "b_la", "b_pull",
            "b_swing_oz"]      # 이 코스 구역(존 안/낮게/높게/바깥/몸쪽 빠짐)에 이 타자가 스윙하는 비율
# 경기 상황: 아웃카운트, 1·2·3루 주자 유무
SIT_F = ["outs", "r1", "r2", "r3"]
# 타구 방향 세분화: 당겨친 뜬공(장타력), 밀어치기, 코스별 당겨치기, 구종 계열별 당겨치기·땅볼
SPRAY_F = ["b_pull_air", "b_oppo", "b_pull_loc", "b_pull_g", "b_gb_g"]
# 직전 공: 구종 계열, 구속 차(완급), 높낮이 차(눈높이)
# 9개를 다 넣으면 오히려 과적합으로 성능이 떨어져서, 검증에서 가장 나았던 3개만 사용
PREV_F = ["prev_grp", "d_velo", "d_z"]
# 타구 방향(SPRAY_F)은 run value 모델에선 효과가 없어서 빼고, 스윙/헛스윙 모델(build_swing.py)에만 사용
FEATURES = PITCH_F + BATTER_F + PREV_F + SIT_F


# ---------------------------------------------------------------- 전처리
def load(path):
    print(f"데이터 읽는 중: {path}")
    df = pd.read_csv(path, usecols=COLS)
    df = add_context_rv(df)      # 필터링 전에 계산해야 투구 순서가 안 끊김
    lookup = {t: g for g, ts in GROUPS.items() for t in ts}
    df["group"] = df["pitch_type"].map(lookup)
    df = df.dropna(subset=["group", "plate_x", "plate_z", "sz_top", "sz_bot",
                           "delta_run_exp", TARGET, "release_speed", "pfx_x", "pfx_z"])
    df["grp"] = df["group"].map(GROUP_CODE)

    # 코스를 '타자 기준'으로: 몸쪽이 항상 +
    df["x_in"] = np.where(df["stand"] == "L", df["plate_x"], -df["plate_x"])
    # 높이를 타자 키에 맞춰 0(무릎)~1(가슴)로
    df["z_norm"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    df["cell"] = cell_id(df["x_in"], df["z_norm"])
    df["oz"] = zone_region(df["x_in"], df["z_norm"])

    # 무브먼트는 '투수 팔 방향'으로 통일 (좌투는 좌우 반전), 인치 단위
    df["pfx_x"] = np.where(df["p_throws"] == "L", -df["pfx_x"], df["pfx_x"]) * 12
    df["pfx_z"] = df["pfx_z"] * 12
    df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
    add_release(df)
    # 주자 칸에는 주자의 선수 ID가 들어 있음 → 비어 있으면 주자 없음
    df["r1"], df["r2"], df["r3"] = (df[c].notna().astype(int) for c in ["on_1b", "on_2b", "on_3b"])
    df["outs"] = df["outs_when_up"]
    df["if_al"] = df["if_fielding_alignment"].map(IF_ALIGN).fillna(0).astype(int)
    df["of_al"] = df["of_fielding_alignment"].map(OF_ALIGN).fillna(0).astype(int)

    df["swing"] = df["description"].isin(SWING).astype(int)
    df["whiff"] = df["description"].isin(WHIFF).astype(int)
    df["in_play"] = (df["description"] == "hit_into_play").astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # 타구 방향: 각도가 음수면 좌측. 우타자는 좌측이 당겨친 것
    spray = np.degrees(np.arctan2(df["hc_x"] - 125.42, 198.27 - df["hc_y"]))
    df["spray_pull"] = np.where(df["stand"] == "R", -spray, spray)   # +면 당긴 쪽, -면 밀어친 쪽 (도)
    df["pull"] = np.where(df["stand"] == "R", spray < -15, spray > 15).astype(float)
    df["oppo"] = np.where(df["stand"] == "R", spray > 15, spray < -15).astype(float)
    no_dir = df["hc_x"].isna() | (df["in_play"] == 0)
    df.loc[no_dir, ["pull", "oppo"]] = np.nan
    # 당겨친 뜬공·라인드라이브(발사각 10~40도) = 홈런이 나오는 타구
    air = df["launch_angle"].between(10, 40)
    df["pull_air"] = (df["pull"] == 1) & air
    df["pull_air"] = df["pull_air"].astype(float).where(~no_dir & df["launch_angle"].notna())
    df["gb"] = (df["launch_angle"] < 10).astype(float).where(
        (df["in_play"] == 1) & df["launch_angle"].notna())
    # 좌우 코스 3구간: 0 바깥쪽, 1 가운데, 2 몸쪽
    df["xloc"] = np.digitize(df["x_in"], [-0.28, 0.28])
    df = add_prev(df)
    print(f"투구 {len(df):,}개")
    return df


def add_prev(df, mask_rate=0.15, seed=0):
    """같은 타석의 바로 앞 공 정보를 붙임. 타석 첫 공이면 비어 있음(NaN)"""
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    g = df.groupby(["game_pk", "at_bat_number"])
    for src, dst in [("grp", "prev_grp"), ("release_speed", "prev_velo"),
                     ("x_in", "prev_x"), ("z_norm", "prev_z"),
                     ("pfx_x", "prev_pfx_x"), ("pfx_z", "prev_pfx_z")]:
        df[dst] = g[src].shift(1)
    # 일부러 15%는 '직전 공 모름'으로 지움 → 추천기에서 직전 공을 안 넣어도 쓸 수 있게
    rng = np.random.default_rng(seed)
    hide = (rng.random(len(df)) < mask_rate) & df["prev_grp"].notna()
    df.loc[hide, ["prev_grp", "prev_velo", "prev_x", "prev_z", "prev_pfx_x", "prev_pfx_z"]] = np.nan
    df["d_velo"] = df["release_speed"] - df["prev_velo"]   # 구속 차 (완급 조절)
    df["d_x"] = df["x_in"] - df["prev_x"]                    # 좌우 코스 이동
    df["d_z"] = df["z_norm"] - df["prev_z"]                  # 높낮이 이동 (눈높이 흔들기)
    df["d_pfx_x"] = df["pfx_x"] - df["prev_pfx_x"]           # 무브먼트 차 (터널링)
    df["d_pfx_z"] = df["pfx_z"] - df["prev_pfx_z"]
    return df


def add_release(df):
    """릴리스 포인트와 홈 진입각(VAA) 계산 (스탯캐스트 50피트 지점 속도·가속도로)"""
    df["rel_z"] = df["release_pos_z"]
    df["rel_side"] = df["release_pos_x"].abs()
    df["ext"] = df["release_extension"]
    vy_f = -np.sqrt(np.maximum(df["vy0"] ** 2 - 2 * df["ay"] * (50 - 17 / 12), 1))
    t = (vy_f - df["vy0"]) / df["ay"]
    vz_f = df["vz0"] + df["az"] * t
    df["vaa"] = np.degrees(np.arctan2(vz_f, -vy_f))            # 음수, 0에 가까울수록 평평
    # 높이·구종 계열이 같은 공끼리 비교한 진입각 → 코스와 겹치지 않는 '투수 특성'
    zbin = pd.cut(df["plate_z"], np.arange(0, 6.01, 0.25))
    exp = df.groupby([df["grp"], zbin], observed=True)["vaa"].transform("mean")
    df["vaa_rel"] = df["vaa"] - exp
    add_body(df)
    return df


def add_body(df, path="player_bio.csv"):
    """투수 키·몸무게 붙이기 + 키 대비 타점/익스텐션 (파일 없으면 비워 둠)"""
    import os
    if os.path.exists(path):
        bio = pd.read_csv(path).set_index("pitcher")
        df["height"] = df["pitcher"].map(bio["height_in"]) / 12
        df["weight"] = df["pitcher"].map(bio["weight_lb"])
    else:
        df["height"] = df["weight"] = np.nan
    df["rel_z_h"] = df["rel_z"] / df["height"]          # 1.0이면 키 높이에서 놓음
    df["ext_h"] = df["ext"] / df["height"]
    return df


def add_context_rv(df):
    """주자·아웃·카운트별 기대 득점표를 데이터로 직접 만들고, 투구 하나의 실점 변화를 계산.

    스탯캐스트의 delta_run_exp는 주자 상황을 제대로 반영 못 함
    (예: 2아웃 3-2 볼넷이 주자 없음 +0.26, 만루 +0.29로 거의 같음).

    기대 득점 RE(상태) = 그 상태에서 이닝이 끝날 때까지 평균 몇 점이 나왔나
    투구의 가치      = 이 공으로 난 점수 + RE(다음 상태) - RE(지금 상태)
    """
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    half = [df["game_pk"], df["inning"], df["inning_topbot"]]
    end_score = df.groupby(half)["post_bat_score"].transform("max")
    runs_rest = end_score - df["bat_score"]                    # 이 공부터 이닝 끝까지 낸 점수
    bases = (df["on_1b"].notna().astype(int) * 1 + df["on_2b"].notna().astype(int) * 2
             + df["on_3b"].notna().astype(int) * 4)
    state = (bases.astype(str) + "_" + df["outs_when_up"].astype(str) + "_"
             + df["balls"].clip(upper=3).astype(str) + df["strikes"].clip(upper=2).astype(str))
    re = runs_rest.groupby(state).mean()
    df["re_now"] = state.map(re)
    nxt = df.groupby(half)["re_now"].shift(-1).fillna(0.0)    # 이닝 마지막 공이면 다음 상태 = 0
    df[TARGET] = (df["post_bat_score"] - df["bat_score"]) + nxt - df["re_now"]
    return df


def zone_region(x_in, z_norm):
    """0 존 안, 1 낮게 빠짐, 2 높게 빠짐, 3 바깥쪽 빠짐, 4 몸쪽 빠짐"""
    x_in, z_norm = np.asarray(x_in), np.asarray(z_norm)
    return np.select([z_norm < 0, z_norm > 1, x_in < -0.83, x_in > 0.83], [1, 2, 3, 4], 0)


def cell_id(x_in, z_norm):
    return np.digitize(x_in, X_EDGES) * 5 + np.digitize(z_norm, Z_EDGES)


def shrunk(df, keys, col, k, prior_keys=None):
    """표본이 적은 타자는 리그 평균 쪽으로 당겨줌 (베이지안 축소)
       결과 = (타자 합계 + k × 리그 평균) / (타자 표본 수 + k)"""
    d = df.dropna(subset=[col])
    agg = d.groupby(keys)[col].agg(["sum", "count"]).reset_index()
    if prior_keys:
        prior = d.groupby(prior_keys)[col].mean().rename("prior").reset_index()
        agg = agg.merge(prior, on=prior_keys)
    else:
        agg["prior"] = d[col].mean()
    agg[col] = (agg["sum"] + k * agg["prior"]) / (agg["count"] + k)
    return agg[keys + [col]]


# ---------------------------------------------------------------- 타자 프로필
def batter_profile(df):
    p = {}
    # 1) 존별 강약: 그 칸으로 온 공의 평균 run value (타자에게 + 면 핫존)
    p["cell"] = shrunk(df, ["batter", "cell"], "delta_run_exp", 40, ["cell"]) \
        .rename(columns={"delta_run_exp": "b_cell_rv"})
    p["cell_league"] = df.groupby("cell")["delta_run_exp"].mean()
    # 코스 구역별 스윙률 (선구안): 저지처럼 낮게 떨어지는 공에 잘 안 속는 타자를 구분
    p["oz"] = shrunk(df, ["batter", "oz"], "swing", 25, ["oz"]).rename(columns={"swing": "b_swing_oz"})

    # 2) 구종 계열별: 헛스윙률(스윙 중), 추격률(존 밖 공에 스윙), 컨택 시 xwOBA
    sw = df[df["swing"] == 1]
    out = df[df["zone"] >= 11]
    contact = df[df["in_play"] == 1]
    g = shrunk(sw, ["batter", "group"], "whiff", 40, ["group"]) \
        .rename(columns={"whiff": "b_whiff"})
    g = g.merge(shrunk(out, ["batter", "group"], "swing", 40, ["group"])
                .rename(columns={"swing": "b_chase"}), on=["batter", "group"], how="outer")
    g = g.merge(shrunk(contact, ["batter", "group"], "estimated_woba_using_speedangle",
                       30, ["group"]).rename(columns={"estimated_woba_using_speedangle": "b_xwoba"}),
                on=["batter", "group"], how="outer")
    # 구종 계열별 당겨치기·땅볼 비율 (변화구에 끌려 나가 땅볼 치는 타자 등)
    g = g.merge(shrunk(contact, ["batter", "group"], "pull", 30, ["group"])
                .rename(columns={"pull": "b_pull_g"}), on=["batter", "group"], how="outer")
    g = g.merge(shrunk(contact, ["batter", "group"], "gb", 30, ["group"])
                .rename(columns={"gb": "b_gb_g"}), on=["batter", "group"], how="outer")
    p["group"] = g

    # 3) 타구 성향 (전체): 평균 타구속도, 발사각, 당겨치기 비율
    o = shrunk(contact, ["batter"], "launch_speed", 30) \
        .rename(columns={"launch_speed": "b_ev"})
    o = o.merge(shrunk(contact, ["batter"], "launch_angle", 30)
                .rename(columns={"launch_angle": "b_la"}), on="batter", how="outer")
    o = o.merge(shrunk(contact, ["batter"], "pull", 30)
                .rename(columns={"pull": "b_pull"}), on="batter", how="outer")
    o = o.merge(shrunk(contact, ["batter"], "pull_air", 40)
                .rename(columns={"pull_air": "b_pull_air"}), on="batter", how="outer")
    o = o.merge(shrunk(contact, ["batter"], "oppo", 30)
                .rename(columns={"oppo": "b_oppo"}), on="batter", how="outer")
    p["overall"] = o

    # 코스별 당겨치기: 바깥쪽 공도 당겨서 치는 타자인가
    p["loc"] = shrunk(contact, ["batter", "xloc"], "pull", 30, ["xloc"]) \
        .rename(columns={"pull": "b_pull_loc"})

    # 프로필이 없는 타자용 리그 평균
    p["defaults"] = {"b_whiff": sw.groupby("group")["whiff"].mean().to_dict(),
                     "b_chase": out.groupby("group")["swing"].mean().to_dict(),
                     "b_xwoba": contact.groupby("group")["estimated_woba_using_speedangle"].mean().to_dict(),
                     "b_ev": contact["launch_speed"].mean(),
                     "b_la": contact["launch_angle"].mean(),
                     "b_pull": contact["pull"].mean(),
                     "b_pull_air": contact["pull_air"].mean(),
                     "b_oppo": contact["oppo"].mean(),
                     "b_pull_g": contact.groupby("group")["pull"].mean().to_dict(),
                     "b_gb_g": contact.groupby("group")["gb"].mean().to_dict(),
                     "b_pull_loc": contact.groupby("xloc")["pull"].mean().to_dict(),
                     "b_swing_oz": df.groupby("oz")["swing"].mean().to_dict()}
    return p


def attach(df, p):
    df = df.merge(p["cell"], on=["batter", "cell"], how="left")
    df["b_cell_rv"] = df["b_cell_rv"].fillna(df["cell"].map(p["cell_league"]))
    df = df.merge(p["group"], on=["batter", "group"], how="left")
    for c in ["b_whiff", "b_chase", "b_xwoba", "b_pull_g", "b_gb_g"]:
        df[c] = df[c].fillna(df["group"].map(p["defaults"][c]))
    df = df.merge(p["overall"], on="batter", how="left")
    for c in ["b_ev", "b_la", "b_pull", "b_pull_air", "b_oppo"]:
        df[c] = df[c].fillna(p["defaults"][c])
    if "oz" not in df:
        df["oz"] = zone_region(df["x_in"], df["z_norm"])
    df = df.merge(p["oz"], on=["batter", "oz"], how="left")
    df["b_swing_oz"] = df["b_swing_oz"].fillna(df["oz"].map(p["defaults"]["b_swing_oz"]))
    df = df.merge(p["loc"], on=["batter", "xloc"], how="left")
    df["b_pull_loc"] = df["b_pull_loc"].fillna(df["xloc"].map(p["defaults"]["b_pull_loc"]))
    return df


# ---------------------------------------------------------------- 모델
def fit(X, y, n=500):
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=63,
                          min_child_samples=300, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1)
    return m.fit(X, y)


def validate(train_df):
    """타자 정보가 예측을 실제로 좋게 만드는지: 마지막 3주를 떼어서 비교"""
    cut = train_df["game_date"].max() - pd.Timedelta(days=21)
    tr, te = train_df[train_df["game_date"] < cut], train_df[train_df["game_date"] >= cut]
    y = TARGET
    base_mse = ((te[y] - tr[y].mean()) ** 2).mean()
    res = {}
    for name, feats in [("상황 정보 없이", PITCH_F + BATTER_F + PREV_F), ("주자·아웃 포함", FEATURES)]:
        m = fit(tr[feats], tr[y], n=300)
        pred = m.predict(te[feats])
        res[name] = (1 - ((te[y] - pred) ** 2).mean() / base_mse, pred)
    print("\n[검증] 마지막 3주 투구로 테스트")
    for name, (r2, _) in res.items():
        print(f"  {name:10s} 설명력(R²) = {r2:.4f}")
    gain = res["주자·아웃 포함"][0] / max(res["상황 정보 없이"][0], 1e-9) - 1
    print(f"  → 주자·아웃 정보를 넣으니 설명력 {gain:+.0%}")
    print("  (투구 하나의 결과는 운이 커서 R²가 원래 작아요. 중요한 건 비교와 아래 보정표)")

    # 보정표: 예측을 10구간으로 나눠, 실제 평균 run value가 같은 순서로 나오나
    pred = res["주자·아웃 포함"][1]
    t = pd.DataFrame({"pred": pred, "actual": te[y].values})
    t["bin"] = pd.qcut(t["pred"], 10, labels=False)
    tab = t.groupby("bin")[["pred", "actual"]].mean() * 100
    print("\n[보정표] 100구당 실점 변화 (예측 vs 실제, -는 투수에게 유리)")
    print(tab.round(2).to_string())


def sequence_evidence(df):
    """실제 데이터로 확인: 직전 공에 따라 '낮은 변화구'의 헛스윙률이 달라지나"""
    low_brk = df[(df["grp"] == 1) & (df["z_norm"] < 0.33) & (df["swing"] == 1)
                 & df["prev_grp"].notna()]
    prev_kind = np.select(
        [(low_brk["prev_grp"] == 0) & (low_brk["prev_z"] > 0.67),
         (low_brk["prev_grp"] == 0),
         (low_brk["prev_grp"] == 1) & (low_brk["prev_z"] < 0.33)],
        ["높은 패스트볼", "다른 패스트볼", "낮은 변화구"], "기타")
    t = low_brk.groupby(prev_kind)["whiff"].agg(["mean", "size"])
    t.columns = ["헛스윙률", "스윙 수"]
    print("\n[실제 데이터] 낮게 던진 변화구의 헛스윙률, 직전 공에 따라")
    t = t.sort_values("헛스윙률", ascending=False)
    t["헛스윙률"] = (t["헛스윙률"] * 100).round(1).astype(str) + "%"
    print(t.to_string())


def arsenal(df):
    """투수별 구종 레퍼토리: 평균 구속·회전·무브먼트, 구사율"""
    a = (df.groupby(["pitcher", "player_name", "p_throws", "pitch_type", "group", "grp"])
           .agg(n=("release_speed", "size"), release_speed=("release_speed", "mean"),
                release_spin_rate=("release_spin_rate", "mean"),
                pfx_x=("pfx_x", "mean"), pfx_z=("pfx_z", "mean"),
                **{c: (c, "mean") for c in RELEASE_F + BODY_F})
           .reset_index())
    a["usage"] = a["n"] / a.groupby("pitcher")["n"].transform("sum")
    return a[(a["n"] >= 30) & (a["usage"] >= 0.04)]


def batter_names(ids):
    try:
        from pybaseball import playerid_reverse_lookup
        t = playerid_reverse_lookup(list(ids), key_type="mlbam")
        t["name"] = (t["name_last"].str.title() + ", " + t["name_first"].str.title())
        return dict(zip(t["key_mlbam"], t["name"]))
    except Exception as e:
        print(f"타자 이름을 못 가져왔어요 ({e}). ID로 검색해야 해요.")
        return {}


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="statcast_2024-04-01_2024-09-30.csv")
    ap.add_argument("--split", default="2024-07-01", help="이 날짜 전 = 프로필용, 이후 = 학습용")
    a = ap.parse_args()

    df = load(a.data)
    sequence_evidence(df)
    first, second = df[df["game_date"] < a.split], df[df["game_date"] >= a.split]

    # 학습: 전반기 프로필 → 후반기 투구
    train_df = attach(second, batter_profile(first))
    validate(train_df)

    print("\n최종 모델 학습 중...")
    model = fit(train_df[FEATURES], train_df[TARGET])

    # 추천할 때는 시즌 전체로 만든 프로필을 사용
    full_profile = batter_profile(df)
    imp = pd.Series(model.feature_importances_, FEATURES).sort_values(ascending=False)
    print("\n[모델이 많이 본 정보 TOP 8]\n" + imp.head(8).to_string())

    zone = df.groupby("batter")[["sz_top", "sz_bot"]].mean()
    names = batter_names(df["batter"].unique())
    with open("matchup_model.pkl", "wb") as f:
        pickle.dump({"model": model, "profile": full_profile, "arsenal": arsenal(df),
                     "batter_names": names, "batter_zone": zone,
                     "batter_stand": df.groupby("batter")["stand"].agg(lambda s: s.mode()[0]),
                     "features": FEATURES}, f)
    print("\n저장 완료: matchup_model.pkl  → 이제 recommend.py로 볼배합을 물어보세요")
