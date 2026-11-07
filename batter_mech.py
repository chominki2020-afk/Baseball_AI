"""
타자 스윙 목표 시뮬레이션 (build_batter_swing.py의 batter_swing.pkl 사용)

그 타자가 실제로 상대한 공들(2025, 코스·카운트·구종 그대로)에서 '타자 스타일'(시즌 평균 스윙 값)만 바꿔
헛스윙 / 인플레이 / 인플레이 xwOBA / 스윙당 xwOBA 를 다시 예측.
  - 배트 스피드를 올리면 스윙 길이도 같이 늘어남 (실제 타자들의 시즌 간 변화에서 배운 비율)
  - 컨택 포인트를 앞으로 옮기면 어택 앵글이 올라가고 스윙 방향은 당기는 쪽으로 같이 움직임 (같은 방식으로 배운 비율)
    → 어택 앵글·방향 축은 '컨택 포인트 때문에 생기는 변화에 더해 추가로' 바꾸는 양
  - 검증(2024→2025 실제로 스윙을 바꾼 타자 286명)에서 모델 반응이 실제보다 큰 만큼(기울기) 줄여서 보여줌
"""
import itertools

import numpy as np
import pandas as pd

# 범위 = 실제 타자들이 한 시즌 사이에 바꾼 폭의 상위 10% 정도까지 (배트 스피드 1.7mph, 어택 앵글 3°, 컨택 3in, 방향 3.8°)
D_BS = [-1, 0, 1, 1.5]        # 배트 스피드 (mph)
D_AA = [-3, -1.5, 0, 1.5, 3]  # 어택 앵글 (°)
D_CY = [-3, 0, 3]             # 컨택 포인트 (인치, +면 더 앞에서)
D_DIR = [-3, 0, 3]            # 어택 디렉션 (°, -면 더 당기는 방향)
FOCUS = {"style": "⭐ 내 스타일에 맞게 (자동)", "xw_sw": "스윙당 xwOBA (종합)", "xwcon": "타구 질 (인플레이 xwOBA)",
         "miss": "헛스윙 줄이기"}
# 이 정도는 넘어야 '의미 있는 변화' — 검증에서 본 예측 불확실성(타자 유형별 어택 앵글 효과 95% 구간 폭 등) 기준
# = 타자 결과가 한 시즌 사이에 원래 출렁이는 폭(표준편차: 스윙당 16.8, 타구 질 38.6, 헛스윙 3.3%p)의 1/3
MEANINGFUL = {"xw_sw": 5.5, "xwcon": 13.0, "miss": 1.1}
# 타자 유형 (배트 스피드 백분위): 컨택형은 컨택을 지키면서, 파워형은 헛스윙을 조금 감수하고 타구 질을
STYLE_RULE = {
    "컨택형": {"target": "xwcon", "max_miss": 0.3, "desc": "헛스윙을 늘리지 않는 범위에서 타구 질 올리기"},
    "균형형": {"target": "xw_sw", "max_miss": None, "desc": "헛스윙과 타구 질을 합친 스윙당 가치 올리기"},
    "파워형": {"target": "xwcon", "max_miss": 1.5, "desc": "헛스윙은 조금(1.5%p까지) 감수하고 타구 질 올리기"},
}


def archetype(rep_row):
    p = rep_row["bat_speed_pct"]
    return "컨택형" if p < 33 else "파워형" if p >= 67 else "균형형"
# 바꾸기 어려운 정도 (한 단위당 감점) — 초점 지표 단위로
COST = {"xw_sw": (1.0, 0.3, 0.4, 0.2), "xwcon": (2.0, 0.6, 0.8, 0.4), "miss": (0.3, 0.1, 0.15, 0.05)}


def resolve(focus, arch):
    """'style'이면 타자 유형에 맞는 (지표, 헛스윙 상한)"""
    if focus == "style":
        r = STYLE_RULE[arch]
        return r["target"], r["max_miss"]
    return focus, None


def predict(m, X):
    if m.get("kind") == "smooth":
        import build_batter_swing as BB
        return BB.predict_smooth(m, X)
    X = X[m["features"]]
    miss = m["miss"].predict_proba(X)[:, 1]
    bip = m["bip"].predict_proba(X)[:, 1]
    xw = m["xwcon"].predict(X)
    return pd.DataFrame({"miss": miss, "bip": bip, "xwcon": xw, "xw_sw": (1 - miss) * bip * xw}, index=X.index)


def load(path="batter_swing.pkl"):
    import os
    import pickle
    if not os.path.exists(path):
        return None
    import build_batter_swing as BB
    return BB.unpack(pickle.load(open(path, "rb")))


def simulate(BS, bid):
    rows = BS["samples"][BS["samples"]["batter"] == bid]
    rep = BS["report"].loc[bid]
    combos = list(itertools.product(D_BS, D_AA, D_CY, D_DIR))
    blocks = []
    k_aa, k_dir = BS.get("k_aa_cy", 0.0), BS.get("k_dir_cy", 0.0)
    for k, (bs, aa, cy, dr) in enumerate(combos):
        x = rows.copy()
        x["m_bat_speed"] += bs
        x["m_swing_length"] += BS["k_len"] * bs
        x["m_attack_angle"] += aa + k_aa * cy
        x["m_contact_y"] += cy
        x["m_attack_direction"] += dr + k_dir * cy
        x["combo"] = k
        blocks.append(x)
    big = pd.concat(blocks, ignore_index=True)
    p = predict(BS["model"], big)
    p["combo"] = big["combo"].values
    p["zb"] = pd.cut(big["z_norm"].values, [-9, 0.33, 0.67, 9], labels=["낮게", "가운데", "높게"])
    agg = p.groupby("combo")[["miss", "bip", "xwcon", "xw_sw"]].mean()
    by_h = p.groupby(["combo", "zb"], observed=True)["xw_sw"].mean().unstack()
    res = pd.DataFrame(combos, columns=["d_bs", "d_aa", "d_cy", "d_dir"])
    res["tot_aa"] = res["d_aa"] + k_aa * res["d_cy"]          # 실제로 바뀌는 총량
    res["tot_dir"] = res["d_dir"] + k_dir * res["d_cy"]
    res = pd.concat([res, agg.reset_index(drop=True)], axis=1)
    base = res[(res[["d_bs", "d_aa", "d_cy", "d_dir"]] == 0).all(axis=1)].iloc[0]
    cal = {k: min(v, 1.0) for k, v in BS["cal"].items() if v == v and v > 0}   # 검증 기울기로 줄이기만 (키우지는 않음)
    res["d_miss"] = (res["miss"] - base["miss"]) * 100 * cal.get("whiff", 1)            # %p
    res["d_bip"] = (res["bip"] - base["bip"]) * 100 * cal.get("bip_rate", 1)
    res["d_xwcon"] = (res["xwcon"] - base["xwcon"]) * 1000 * cal.get("xwcon", 1)         # xwOBA 포인트 (0.001)
    res["d_xw_sw"] = (res["xw_sw"] - base["xw_sw"]) * 1000 * cal.get("xw_sw", 1)
    bh = by_h.loc[base.name] if base.name in by_h.index else None
    for z in by_h.columns:
        res["h_" + z] = ((by_h[z].reindex(range(len(res))).values - by_h.loc[base.name, z]) * 1000 * cal.get("xw_sw", 1))
    # 현실성: 바꾼 뒤 시즌 평균이 MLB 타자 범위(2.5~97.5%) 안 (이미 밖이면 지금 값까지는 허용)
    cur = {"bat_speed": rep["bat_speed"], "attack_angle": rep["attack_angle"], "contact_y": rep["contact_y"],
           "attack_direction": rep["attack_direction"], "swing_length": rep["swing_length"]}
    ok = np.ones(len(res), bool)
    for col, dcol, mult in (("bat_speed", "d_bs", 1), ("attack_angle", "tot_aa", 1), ("contact_y", "d_cy", 1),
                            ("attack_direction", "tot_dir", 1), ("swing_length", "d_bs", BS["k_len"])):
        lo, hi = BS["bounds"][col]
        lo, hi = min(lo, cur[col]), max(hi, cur[col])
        v = cur[col] + res[dcol] * mult
        ok &= ((v >= lo - 1e-9) & (v <= hi + 1e-9)).to_numpy()
    res["feasible"] = ok
    res["new_bs"] = cur["bat_speed"] + res["d_bs"]
    res["new_len"] = cur["swing_length"] + res["d_bs"] * BS["k_len"]
    return res


def best(res, focus, only=None, protect=True, arch="균형형"):
    """focus: 'style' | 'xw_sw' | 'xwcon' | 'miss'.  only: 바꿀 축 제한 (예: ['d_aa'])
    결과에 meaningful 열: 이득이 MEANINGFUL 기준을 넘는지"""
    focus, max_miss = resolve(focus, arch)
    r = res[res["feasible"]].copy()
    if max_miss is not None:
        r = r[r["d_miss"] <= max_miss]
    axes = ["d_bs", "d_aa", "d_cy", "d_dir"]
    if only is not None:
        r = r[(r[[a for a in axes if a not in only]] == 0).all(axis=1)]
    gain = -r["d_miss"] if focus == "miss" else r["d_" + focus]
    c = COST[focus]
    r["score"] = gain - sum(c[i] * r[a].abs() for i, a in enumerate(axes))
    if protect and focus != "xw_sw":
        r = r[r["d_xw_sw"] >= -1.0]               # 다른 걸 살리다 종합 가치가 1포인트 넘게 떨어지면 제외
    r["gain"] = gain.reindex(r.index)
    r["meaningful"] = r["gain"] >= MEANINGFUL[focus]
    r["focus_used"] = focus
    return r.sort_values("score", ascending=False)


def words(row, k_len):
    """실제로 바뀌는 총량으로 설명 (컨택 포인트를 옮기면 어택 앵글·방향도 자연히 같이 움직이는 것 포함)"""
    w = []
    if row["d_bs"]:
        w.append(f"배트 스피드 {row['d_bs']:+g}mph (스윙 길이 {row['d_bs'] * k_len * 30.48:+.0f}cm 같이)")
    if row["d_cy"]:
        w.append(f"컨택 포인트 {abs(row['d_cy']) * 2.54:.0f}cm {'앞으로' if row['d_cy'] > 0 else '뒤로'}")
    aa = row["tot_aa"] if "tot_aa" in row else row["d_aa"]
    dr = row["tot_dir"] if "tot_dir" in row else row["d_dir"]
    tag = lambda own: "" if not row["d_cy"] else (" (컨택 포인트 따라 자연히)" if own == 0 else " (컨택 포인트 이동분 포함)")
    if abs(aa) >= 0.3:
        w.append(f"어택 앵글 {aa:+.1f}° ({'더 올려 치기' if aa > 0 else '더 평평하게'}){tag(row['d_aa'])}")
    elif row["d_cy"] and row["d_aa"]:
        w.append("어택 앵글은 지금처럼 유지 (앞뒤로 옮기면 자연히 변하는 걸 잡아주기)")
    if abs(dr) >= 0.3:
        w.append(f"스윙 방향 {abs(dr):.1f}° {'당기는 쪽으로' if dr < 0 else '밀어치는 쪽으로'}{tag(row['d_dir'])}")
    elif row["d_cy"] and row["d_dir"]:
        w.append("스윙 방향은 지금처럼 유지 (앞뒤로 옮기면 자연히 변하는 걸 잡아주기)")
    return " · ".join(w) or "지금 스윙 유지"
