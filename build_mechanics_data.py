"""
투수 개선(메카닉) 분석용 데이터 만들기

  1) pitcher_report.pkl : 투수×구종별 지표(구속·회전·무브먼트·익스텐션·진입각·헛스윙·실점 가치)와
                          같은 구종 리그 투수들 사이 백분위, 투수별 '제일 잘 던지는 공'
  2) pitch_samples.pkl  : 투수별 실제 투구 샘플(구종당 최대 60개) — 스윙/헛스윙 모델에 그대로 넣을 수 있는 형태
                          (코스·카운트·상황·직전 공·상대 타자 성향까지 실제 그대로)

실행: python build_mechanics_data.py   (2분 정도)
"""
import pickle

import numpy as np
import pandas as pd

import build_matchup as B
import build_swing as W

MIN_PITCHER = 300      # 이 투구 수 이상 투수만
MIN_TYPE = 40          # 구종 최소 투구 수
PER_TYPE = 60          # 구종당 샘플 수


def report(df):
    g = df.groupby(["pitcher", "player_name", "p_throws", "pitch_type"])
    t = g.agg(n=("release_speed", "size"), velo=("release_speed", "mean"), spin=("release_spin_rate", "mean"),
              ivb=("pfx_z", "mean"), hb=("pfx_x", "mean"), ext=("ext", "mean"), rel_z=("rel_z", "mean"),
              rel_side=("rel_side", "mean"), vaa_rel=("vaa_rel", "mean"), swings=("swing", "sum"),
              whiffs=("whiff", "sum"), rv=(B.TARGET, "mean")).reset_index()
    t["usage"] = t["n"] / t.groupby("pitcher")["n"].transform("sum")
    t["whiff_pct"] = t["whiffs"] / t["swings"].clip(lower=1)        # 스윙했을 때 헛스윙
    t["swstr"] = t["whiffs"] / t["n"]                                # 투구당 헛스윙 (SwStr%)
    t["rv100"] = t["rv"] * 100                                       # 100구당 실점 (낮을수록 좋음)
    ok = t["n"] >= 100
    for c in ["velo", "spin", "ivb", "hb", "ext", "vaa_rel", "whiff_pct", "swstr"]:
        t[c + "_pct"] = np.nan
        t.loc[ok, c + "_pct"] = t[ok].groupby("pitch_type")[c].rank(pct=True) * 100
    t["rv100_pct"] = np.nan
    t.loc[ok, "rv100_pct"] = (1 - t[ok].groupby("pitch_type")["rv100"].rank(pct=True)) * 100 + 100 / ok.sum()
    t["rv100_pct"] = t["rv100_pct"].clip(upper=100)
    # 제일 잘 던지는 공: 많이 던진 구종(100구 이상, 구사율 8% 이상) 중 실점 가치 백분위가 가장 좋은 공
    cand = t[ok & (t["usage"] >= 0.08)]
    best = cand.loc[cand.groupby("pitcher")["rv100_pct"].idxmax(), ["pitcher", "pitch_type"]]
    t["is_best"] = t.set_index(["pitcher", "pitch_type"]).index.isin(best.set_index(["pitcher", "pitch_type"]).index)
    return t


if __name__ == "__main__":
    df = B.load("statcast_2024-04-01_2024-09-30.csv")
    rep = report(df)
    prof = B.batter_profile(df)
    big = df.groupby("pitcher")["pitcher"].transform("size") >= MIN_PITCHER
    d = df[big]
    d = d[d.groupby(["pitcher", "pitch_type"])["pitch_type"].transform("size") >= MIN_TYPE]
    samp = (d.groupby(["pitcher", "pitch_type"], group_keys=False)
              .apply(lambda x: x.sample(min(len(x), PER_TYPE), random_state=0)))
    samp = B.attach(samp, prof)
    keep = ["pitcher", "pitch_type", "stand"] + [c for c in W.FULL_F if c not in ("pitcher",)]
    samp = samp[keep].copy()
    num = samp.select_dtypes("number").columns.drop("pitcher")
    samp[num] = samp[num].astype(np.float32)
    with open("pitcher_report.pkl", "wb") as f:
        pickle.dump(rep, f)
    with open("pitch_samples.pkl", "wb") as f:
        pickle.dump(samp.reset_index(drop=True), f)
    print(f"리포트: 투수 {rep['pitcher'].nunique()}명 / 샘플: 투수 {samp['pitcher'].nunique()}명, {len(samp):,}구")
    s = rep[rep["player_name"].isin(["Skenes, Paul", "Hader, Josh", "Sale, Chris"])]
    print(s[["player_name", "pitch_type", "n", "usage", "velo", "swstr", "rv100", "rv100_pct", "is_best"]]
          .round(3).to_string(index=False))
