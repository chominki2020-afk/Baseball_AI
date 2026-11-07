"""
투수 개선(메카닉) 시뮬레이션

질문: 이 투수가 릴리스(타점 높이·몸에서 옆으로·익스텐션)를 바꾸면 헛스윙이 늘어날까?

방법
  - 그 투수가 2024년에 실제로 던진 공들(코스·카운트·상황·상대 타자 그대로)을 가져와서
  - 릴리스 관련 값만 바꿔 스윙/헛스윙 모델에 다시 넣음
      타점 높이(rel_z), 옆 위치(rel_side), 익스텐션(ext)  → 바꾼 만큼
      진입각(vaa_rel)                                     → 바뀐 릴리스로 궤적을 다시 계산해서 생기는 차이만큼
  - 투구당 헛스윙(SwStr% = 스윙 확률 × 헛스윙 확률)을 구종별로 비교
  - 구속·회전·무브먼트는 그대로라고 가정 (실제로는 릴리스를 바꾸면 이것들도 변할 수 있음)
"""
import itertools

import numpy as np
import pandas as pd

import trajectory as T

DH = [-0.5, -0.25, 0.0, 0.25, 0.5]        # 타점 높이 변화 (피트)
DS = [-0.5, -0.25, 0.0, 0.25, 0.5]        # 옆 위치 변화 (+면 몸 바깥쪽으로 = 사이드암 쪽)
DE = [-0.25, 0.0, 0.25, 0.5]              # 익스텐션 변화 (+면 더 앞에서 놓음)


RANGE_F = ("rel_z", "rel_side", "ext")     # 현실성 검사: 실제 투수들이 쓰는 릴리스 범위 안인가
MAIN_USAGE = 0.05                           # 이 구사율 이상 구종만 검사·합산


def league_bounds(report, hand):
    """구종별 실제 MLB 투수 범위 (2.5~97.5%) — 이 범위를 벗어나는 변화는 '현실적이지 않음'으로 제외"""
    r = report[(report["n"] >= 100) & (report["p_throws"] == hand)]
    return {t: {c: (g[c].quantile(0.025), g[c].quantile(0.975)) for c in RANGE_F}
            for t, g in r.groupby("pitch_type")}


GROUP = {"FF": "fb", "SI": "fb", "FC": "fb", "SL": "brk", "ST": "brk", "CU": "brk", "KC": "brk", "SV": "brk",
         "CS": "brk", "CH": "off", "FS": "off", "FO": "off", "SC": "off"}
MOVE_Y = {"ivb": "pfx_z", "hb": "pfx_x", "velo": "release_speed"}     # 학습한 변화 → 모델 입력 열


def load_move(path="release_move.pkl"):
    """릴리스 1피트당 무브먼트·구속 변화 계수.
    시즌 간 변화(A)와 경기 간 변화(B) 두 방법이 '같은 방향'일 때만 쓰고, 크기는 둘 중 작은 쪽(보수적)."""
    import os
    import pickle
    if not os.path.exists(path):
        return None
    r = pickle.load(open(path, "rb"))
    sea = r[r["method"] == "season"].set_index(["type", "y"])
    out = {}
    for (t, y), g in r.groupby(["type", "y"]):
        if y not in MOVE_Y:
            continue
        g = g.set_index("method")
        if "game" not in g.index:
            continue
        # 시즌 간 표본이 적은 구종(FS, KC 등)은 같은 계열의 시즌 간 계수로 방향 확인
        s_key = (t, y) if (t, y) in sea.index else (GROUP.get(t, t), y)
        coef = []
        for x in ("rel_z", "rel_side", "ext"):
            b1 = g.loc["game", "b_" + x]
            b2 = sea.loc[s_key, "b_" + x] if s_key in sea.index else np.nan
            coef.append(np.sign(b1) * min(abs(b1), abs(b2)) if np.isfinite(b2) and np.sign(b1) == np.sign(b2) else 0.0)
        out[(t, y)] = np.array(coef)
    return out


def move_delta(move, t, dh, ds, de):
    """구종 t의 릴리스를 바꿨을 때 {ivb(인치), hb(인치, -면 팔 쪽), velo(mph)} 변화"""
    d = {y: 0.0 for y in MOVE_Y}
    if move is None:
        return d
    for y in MOVE_Y:
        c = move.get((t, y))
        if c is None:                                   # 구종별 표본이 적으면 계열(패스트볼/변화구/오프스피드) 계수
            c = move.get((GROUP.get(t, "fb"), y), np.zeros(3))
        d[y] = float(c @ np.array([dh, ds, de]))
    return d


def vaa_shift(phys_row, dh, ds, de, d_ivb=0.0):
    """릴리스를 바꿨을 때 그 구종 평균 코스에서 진입각이 몇 도 바뀌나 (+면 더 평평).
    d_ivb: 수직 무브먼트 변화(인치) — 40피트 지점부터 홈까지 생기는 휨이라 수직 가속도로 바꿔서 반영"""
    tgt = (phys_row["plate_x"], phys_row["plate_z"])
    a0 = T.approach_angles(T.path(phys_row, *tgt))[0]
    q = T.with_release(phys_row, dh, ds, de)
    if d_ivb:
        tr = T.path(q, *tgt)
        t40 = tr[0][-1] - np.interp(40.0, tr[2][::-1], tr[0][::-1])     # 40피트 → 홈까지 시간
        q["az"] = q["az"] + 2 * (d_ivb / 12) / t40 ** 2
    a1 = T.approach_angles(T.path(q, *tgt))[0]
    return a1 - a0


def load_personal(path="personal_move.pkl"):
    """투수 본인의 릴리스→무브 계수 {(pitcher, type): {"ivb": array3, "hb": array3}} (build_personal.py)"""
    import os
    import pickle
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, "rb"))["app"]


def fix_rel(b, S, fb_shift):
    """자기 패스트볼 대비 피처 다시 계산 (v2 모델일 때). fb_shift = 주력 패스트볼의 (구속, 수직, 모델 가로) 변화"""
    if "rel" not in S:
        return b
    b["fb_dv"] = b["release_speed"] - (b["fb_velo_ref"] + fb_shift[0])
    b["fb_divb"] = b["pfx_z"] - (b["fb_ivb_ref"] + fb_shift[1])
    b["fb_dhb"] = b["pfx_x"] - (b["fb_hb_ref"] + fb_shift[2])
    return b


def simulate(S, samples, phys, report, pid, hand, move=None, personal=None):
    """모든 (타점, 옆, 익스텐션) 조합에 대해 구종별 예상 SwStr% 계산 → 긴 표

    검증(validate_release.py) 결과에 따라 '릴리스 → 무브먼트 변화 → 헛스윙' 경로로만 계산:
      - 스윙 모델에 릴리스 값(타점·옆·익스텐션)을 직접 바꿔 넣으면 실제 2025 변화와 상관 -0.06 (틀림)
        → 모델이 '타점 높은 투수가 원래 잘 던지더라'(투수 간 차이)를 배운 것이라 인과가 아님
      - 공 모양(무브·구속)이 바뀌었을 때의 모델 반응은 실제와 상관 +0.28 (검증됨)
    move: load_move() 결과 (릴리스 1피트당 무브먼트 변화). 없으면 릴리스를 바꿔도 변화 없음
    personal: load_personal() 결과. 있으면 그 투수 본인의 릴리스→무브 관계를 씀 (구속은 리그 관계)"""
    rep = report[report["pitcher"] == pid].set_index("pitch_type")
    main = rep.index[rep["usage"] >= MAIN_USAGE]
    rows = samples[(samples["pitcher"] == pid) & samples["pitch_type"].isin(main)]
    types = sorted(rows["pitch_type"].unique())
    bounds = league_bounds(report, hand)
    combos = list(itertools.product(DH, DS, DE))
    mv, src = {}, {}
    for t in types:
        pc = personal.get((pid, t)) if personal else None
        src[t] = "개인" if pc is not None else "리그"
        mv[t] = {}
        for c in combos:
            m = move_delta(move, t, *c)
            if pc is not None:
                m["ivb"], m["hb"] = float(pc["ivb"] @ np.array(c)), float(pc["hb"] @ np.array(c))
            mv[t][c] = m
    fb_t = rows["main_fb"].iloc[0] if "main_fb" in rows and len(rows) else None
    # 구종별 진입각 변화 미리 계산
    shift = {}
    for t in types:
        if t in phys.index:
            p = phys.loc[t].to_dict()
            shift[t] = {c: vaa_shift(p, 0, 0, 0, d_ivb=mv[t][c]["ivb"]) for c in combos}   # 무브 변화로 생기는 진입각만
        else:
            shift[t] = {c: 0.0 for c in combos}
    blocks = []
    for k, c in enumerate(combos):
        b = rows.copy()                       # 릴리스 값 자체는 그대로 둠 (위 설명)
        b["vaa_rel"] += b["pitch_type"].map({t: shift[t][c] for t in types})
        if move is not None:
            for y, col in MOVE_Y.items():
                b[col] += b["pitch_type"].map({t: mv[t][c][y] for t in types})
        fs = mv[fb_t][c] if fb_t in mv else {"velo": 0.0, "ivb": 0.0, "hb": 0.0}
        b = fix_rel(b, S, (fs["velo"], fs["ivb"], fs["hb"]))
        b["combo"] = k
        blocks.append(b)
    big = pd.concat(blocks, ignore_index=True)
    X = big[S["features"]]
    big["swstr"] = S["swing"].predict_proba(X)[:, 1] * S["whiff"].predict_proba(X)[:, 1]
    out = big.groupby(["combo", "pitch_type"])["swstr"].mean().unstack()
    res = []
    for k, c in enumerate(combos):
        feasible, why = True, []
        for t in types:
            if t not in rep.index or t not in bounds:
                continue
            cur = {"rel_z": rep.loc[t, "rel_z"], "rel_side": rep.loc[t, "rel_side"], "ext": rep.loc[t, "ext"]}
            new = {"rel_z": cur["rel_z"] + c[0], "rel_side": cur["rel_side"] + c[1], "ext": cur["ext"] + c[2]}
            for col, v in new.items():
                lo, hi = bounds[t][col]
                lo, hi = min(lo, cur[col]), max(hi, cur[col])     # 이미 범위 밖인 투수: 지금 위치까지는 OK, 더 멀어지는 것만 제외
                if not (lo - 1e-9 <= v <= hi + 1e-9):
                    feasible = False
                    why.append(col)
        r = {"dh": c[0], "ds": c[1], "de": c[2], "feasible": feasible, "why": ",".join(sorted(set(why)))}
        for t in types:
            r[t] = out.loc[k, t]
            r["vaa_" + t] = shift[t][c]
            for y in MOVE_Y:
                r[f"{y}_{t}"] = mv[t][c][y]
        res.append(r)
    res = pd.DataFrame(res)
    base = res[(res["dh"] == 0) & (res["ds"] == 0) & (res["de"] == 0)].iloc[0]
    for t in types:
        res["d_" + t] = (res[t] - base[t]) * 100                    # %p 변화
    usage = rep["usage"].reindex(types).fillna(0)
    res["d_all"] = sum(res["d_" + t] * usage[t] for t in types) / usage.sum()
    res.attrs["src"] = src
    return res, types, usage


def best_plan(res, focus, types, max_step=None, penalty=0.2, protect=True):
    """기준(focus: 구종 코드 또는 'all')의 헛스윙 증가가 가장 큰 현실적인 조합.
    - 모델 노이즈에 덜 휘둘리게, 바꾸는 양이 클수록 조금씩 감점 (penalty %p per 피트)
    - protect: 한 구종을 살리려다 전체(구사율 가중)가 손해 보는 조합은 제외"""
    col = "d_all" if focus == "all" else "d_" + focus
    r = res[res["feasible"]].copy()
    if max_step is not None:
        r = r[(r[["dh", "ds", "de"]].abs() <= max_step).all(axis=1)]
    if protect and focus != "all":
        r = r[r["d_all"] >= -0.05]
    r["score"] = r[col] - penalty * r[["dh", "ds", "de"]].abs().sum(axis=1)
    return r.sort_values("score", ascending=False)


# ---------------------------------------------------------------- 공 모양 목표
D_IVB = [-4, -2, 0, 2, 4]         # 수직 무브 변화 (인치, +면 덜 떨어짐/더 떠오름)
D_ARM = [-4, -2, 0, 2, 4]         # 가로 무브 변화 (인치, +면 팔 쪽으로 더 휨)
D_VELO = [-1, 0, 1, 2]            # 구속 변화 (mph)


def shape_bounds(report, hand):
    """구종별 실제 MLB 투수 공 모양 범위 (2.5~97.5%)"""
    r = report[(report["n"] >= 100) & (report["p_throws"] == hand)]
    return {t: {c: (g[c].quantile(0.025), g[c].quantile(0.975)) for c in ("ivb", "hb", "velo")}
            for t, g in r.groupby("pitch_type")}


def shape_simulate(S, samples, phys, report, pid, hand):
    """구종마다 공 모양(수직 무브·가로 무브·구속)을 바꾸면 그 구종 헛스윙이 얼마나 변하나.
    그 투수의 2024 실제 투구(코스·카운트·타자 그대로)에서 그 구종의 공 모양만 바꿈 (검증된 부분)"""
    rep = report[report["pitcher"] == pid].set_index("pitch_type")
    main = rep.index[rep["usage"] >= MAIN_USAGE]
    rows = samples[(samples["pitcher"] == pid) & samples["pitch_type"].isin(main)]
    types = sorted(rows["pitch_type"].unique())
    bounds = shape_bounds(report, hand)
    combos = list(itertools.product(D_IVB, D_ARM, D_VELO))
    blocks, meta = [], []
    rel = "rel" in S
    fb_t = rows["main_fb"].iloc[0] if rel and len(rows) else None
    for t in types:
        # v2(패스트볼 대비 피처) 모델에서 주력 패스트볼을 바꾸면 다른 구종도 '대비 값'이 바뀌어 같이 영향 받음
        base = rows if (rel and t == fb_t) else rows[rows["pitch_type"] == t]
        own = (base["pitch_type"] == t).to_numpy()
        p = phys.loc[t].to_dict() if t in phys.index else None
        vaa = {iv: (vaa_shift(p, 0, 0, 0, d_ivb=iv) if p is not None else 0.0) for iv in D_IVB}
        for k, (iv, arm, v) in enumerate(combos):
            b = base.copy()
            b.loc[own, "pfx_z"] += iv
            b.loc[own, "pfx_x"] -= arm           # 모델 좌표는 -가 팔 쪽
            b.loc[own, "release_speed"] += v
            b.loc[own, "vaa_rel"] += vaa[iv]
            b = fix_rel(b, S, (v, iv, -arm) if t == fb_t else (0.0, 0.0, 0.0))
            b["changed"] = t
            b["combo"] = k
            blocks.append(b)
            cur = {"ivb": rep.loc[t, "ivb"], "hb": rep.loc[t, "hb"], "velo": rep.loc[t, "velo"]}
            new = {"ivb": cur["ivb"] + iv, "hb": cur["hb"] - arm, "velo": cur["velo"] + v}
            ok = True
            if t in bounds:
                for c_, val in new.items():
                    lo, hi = bounds[t][c_]
                    lo, hi = min(lo, cur[c_]), max(hi, cur[c_])
                    ok &= lo - 1e-9 <= val <= hi + 1e-9
            meta.append({"pitch_type": t, "combo": k, "d_ivb": iv, "d_arm": arm, "d_velo": v, "feasible": ok,
                         "vaa": vaa[iv], "new_ivb": new["ivb"], "new_arm": -new["hb"], "new_velo": new["velo"]})
    big = pd.concat(blocks, ignore_index=True)
    X = big[S["features"]]
    big["swstr"] = S["swing"].predict_proba(X)[:, 1] * S["whiff"].predict_proba(X)[:, 1]
    pt = big.groupby(["changed", "combo", "pitch_type"])["swstr"].mean().unstack()      # 바꾼 구종 × 조합 → 구종별
    base_all = rows.copy()
    X0 = base_all[S["features"]]
    b0 = pd.Series(S["swing"].predict_proba(X0)[:, 1] * S["whiff"].predict_proba(X0)[:, 1],
                   index=base_all.index).groupby(base_all["pitch_type"]).mean()
    pt = pt.apply(lambda col: col.fillna(b0[col.name]))                                  # 영향 안 받은 구종 = 기준값
    dpt = (pt - b0) * 100
    usage = rep["usage"].reindex(types).fillna(0)
    res = pd.DataFrame(meta)
    idx = pd.MultiIndex.from_frame(res[["pitch_type", "combo"]])
    res["swstr"] = [pt.loc[(t, k), t] for t, k in idx]
    res["d"] = [dpt.loc[(t, k), t] for t, k in idx]
    res["d_all"] = [(dpt.loc[(t, k)] * usage).sum() / usage.sum() for t, k in idx]
    for u in types:
        res["x_" + u] = [dpt.loc[(t, k), u] for t, k in idx]
    res["d_others"] = [((dpt.loc[(t, k)].drop(t) * usage.drop(t)).sum() / max(usage.drop(t).sum(), 1e-9))
                       for t, k in idx]
    return res, types


def best_shape(res, t, velo=None, per_inch=0.1, per_mph=0.3, others_min=-0.2):
    """구종 t의 가장 효과적인 공 모양 변화 (바꾸는 양이 클수록 감점: 인치당 per_inch, mph당 per_mph %p)
    others_min: 주력 패스트볼을 바꿀 때 다른 구종들(구사율 가중)이 이보다 더 손해 보는 조합은 제외"""
    r = res[(res["pitch_type"] == t) & res["feasible"]].copy()
    if "d_others" in r:
        r = r[r["d_others"] >= others_min]
    if velo is not None:
        r = r[r["d_velo"] == velo]
    r["score"] = r["d"] - per_inch * (r["d_ivb"].abs() + r["d_arm"].abs()) - per_mph * r["d_velo"].abs()
    return r.sort_values("score", ascending=False)
