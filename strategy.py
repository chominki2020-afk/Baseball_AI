"""
수비 위치 / 타구 방향 전략 계산 (bip_model.pkl 사용)
"""
import numpy as np
import pandas as pd

PULL, OPPO = 15, -15          # 타구 방향 구분 (도): +15 넘으면 당긴 쪽, -15 미만이면 밀어친 쪽
TYPES = [("땅볼", -90, 10), ("라인드라이브", 10, 25), ("뜬공", 25, 50)]


def batter_bip(Bm, bid, stand, min_n=40):
    b = Bm["batter_bip"]
    mine = b[b["batter"] == bid]
    if len(mine) >= min_n:
        return mine, False
    return b[b["stand"] == stand], True            # 표본 부족 → 같은 타석 방향 리그 평균


def spray_summary(bip):
    """땅볼 / 공중 타구 각각 당김·가운데·밀어침 비율"""
    out = {}
    for name, cond in [("땅볼", bip["launch_angle"] < 10), ("공중 타구", bip["launch_angle"] >= 10)]:
        s = bip.loc[cond, "spray_pull"]
        out[name] = {"당김": (s > PULL).mean(), "가운데": s.between(OPPO, PULL).mean(),
                     "밀어침": (s < OPPO).mean(), "n": len(s)}
    return out


def _predict(Bm, X):
    return Bm["model"].predict(X[Bm["features"]])


def align_values(Bm, bip):
    """이 타자의 실제 타구들을 수비 배치마다 넣어 봤을 때 평균 wOBA"""
    rows = []
    for if_al in (0, 1, 2):
        for of_al in (0, 1):
            X = bip.assign(if_al=if_al, of_al=of_al)
            rows.append({"if_al": if_al, "of_al": of_al, "woba": _predict(Bm, X).mean()})
    t = pd.DataFrame(rows)
    std = t.loc[(t["if_al"] == 0) & (t["of_al"] == 0), "woba"].iloc[0]
    t["runs100"] = (t["woba"] - std) / Bm["scale"] * 100      # 정상 수비 대비, 타구 100개당 실점 차이
    return t.sort_values("woba")


def direction_values(Bm, bip, if_al, of_al):
    """타구 종류별로, 같은 속도·각도로 당겼을 때 / 가운데 / 밀었을 때 평균 wOBA"""
    rows = []
    for name, lo, hi in TYPES:
        part = bip[bip["launch_angle"].between(lo, hi, inclusive="left")]
        if len(part) < 10:
            continue
        r = {"타구": name, "비율": len(part) / len(bip)}
        for label, ang in [("당겨서", 25), ("가운데로", 0), ("밀어서", -25)]:
            X = part.assign(spray_pull=ang, if_al=if_al, of_al=of_al)
            r[label] = _predict(Bm, X).mean()
        r["실제 방향"] = part["spray_pull"].mean()
        rows.append(r)
    return pd.DataFrame(rows)
