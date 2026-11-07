"""
볼배합 추천기: 이 투수가 이 타자에게, 이 카운트에서 무엇을 어디로 던져야 하나

방법:
  1) 투수의 실제 구종(평균 구속·회전·무브먼트)을 가져옴
  2) 각 구종 × 코스(몸쪽~바깥쪽, 낮게~높게) 조합을 전부 만들어 봄
  3) 투수는 정확히 그 자리에 못 던지므로, 목표 지점 주변으로 흩어지게 여러 번 던진 것으로 가정
  4) 매치업 모델로 각 조합의 기대 run value를 계산 → 낮을수록 투수에게 유리

실행 예:
  python recommend.py --pitcher Skenes --batter Judge --count 1-2
  python recommend.py --pitcher "Hader" --batter "Ohtani" --count 0-0 --stand L
"""
import argparse
import os
import pickle
import sys
import unicodedata

import numpy as np
import pandas as pd

X_TARGETS = np.round(np.arange(-1.3, 1.31, 0.2), 2)     # 바깥쪽(-) ~ 몸쪽(+), 피트
Z_TARGETS = np.round(np.arange(-0.5, 1.51, 0.125), 3)   # 무릎(0) ~ 가슴(1)
ZONE_X, ZONE_Z = (-0.83, 0.83), (0.0, 1.0)


def plain(text):
    """악센트 제거: Arráez → arraez"""
    t = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def find(names, query, what):
    q = plain(query)
    hits = {k: v for k, v in names.items() if q in plain(v)}
    if len(hits) == 1:
        return next(iter(hits.items()))
    if not hits:
        sys.exit(f"{what} '{query}'를 못 찾았어요. 성(last name)으로 검색해 보세요.")
    exact = {k: v for k, v in hits.items() if plain(v).startswith(q)}
    if len(exact) == 1:
        return next(iter(exact.items()))
    print(f"{what} 후보가 여러 명이에요. 더 정확히 써 주세요:")
    for v in sorted(set(hits.values()))[:15]:
        print("  -", v)
    sys.exit()


def describe(x, z):
    side = ("바깥쪽 빠지게" if x < -0.83 else "바깥쪽" if x < -0.28 else
            "가운데" if x <= 0.28 else "몸쪽" if x <= 0.83 else "몸쪽 빠지게")
    height = ("낮게 떨어지게" if z < 0 else "낮게" if z < 0.33 else
              "가운데 높이" if z <= 0.67 else "높게" if z <= 1 else "높게 빠지게")
    return f"{side}, {height}"


SPOT_X = {"몸쪽 빠지게": 1.0, "바깥쪽 빠지게": -1.0, "몸쪽": 0.55, "바깥쪽": -0.55}
SPOT_Z = {"높게 빠지게": 1.2, "낮게 떨어지게": -0.2, "높게": 0.85, "낮게": 0.15}


def parse_spot(text):
    """'높게 몸쪽', '낮게 떨어지게 바깥쪽' 같은 말을 좌표로"""
    text = (text or "").replace("-", " ").replace(",", " ")
    x = next((v for k, v in SPOT_X.items() if k in text), 0.0)
    z = next((v for k, v in SPOT_Z.items() if k in text), 0.5)
    return x, z


def load_embed(path="embed_model.npz"):
    """선수 임베딩 신경망 가중치 (build_embed.py save). 없으면 None → LightGBM만 사용"""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ("eb", "ep", "proj_w", "proj_b", "f0_w", "f0_b", "f2_w", "f2_b", "g0_w", "g0_b",
                              "g2_w", "g2_b", "mu", "sd", "rv_sd", "b_ids", "b_idx", "p_ids", "p_idx")}


def embed_swing(W, g, pitcher_id, batter_id):
    """신경망+임베딩의 스윙·헛스윙 확률. 처음 보는 선수는 임베딩 0 (= 리그 평균)"""
    from build_embed import np_forward, prep
    X, _ = prep(g, {"mu": W["mu"], "sd": W["sd"]})
    bi = dict(zip(W["b_ids"].tolist(), W["b_idx"].tolist())).get(int(batter_id), 0)
    pi = dict(zip(W["p_ids"].tolist(), W["p_idx"].tolist())).get(int(pitcher_id), 0)
    n = len(g)
    ps, pw, _ = np_forward(W, X, np.full(n, bi), np.full(n, pi))
    return ps, pw


def simulate(M, pitcher_id, batter_id, balls, strikes, stand, command_ft,
             prev=None, S=None, sit=(0, 0, 0, 0), n_noise=40, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    ars = M["arsenal"][M["arsenal"]["pitcher"] == pitcher_id]
    prof = M["profile"]
    zone = M["batter_zone"].loc[batter_id]
    zh = zone["sz_top"] - zone["sz_bot"]

    # 목표 지점 격자 × 제구 오차
    tx, tz = np.meshgrid(X_TARGETS, Z_TARGETS, indexing="ij")
    tx, tz = tx.ravel(), tz.ravel()
    ex = rng.normal(0, command_ft, n_noise)
    ez = rng.normal(0, command_ft / zh, n_noise)
    grid = pd.DataFrame({"tx": np.repeat(tx, n_noise), "tz": np.repeat(tz, n_noise),
                         "x_in": np.repeat(tx, n_noise) + np.tile(ex, len(tx)),
                         "z_norm": np.repeat(tz, n_noise) + np.tile(ez, len(tz))})
    grid["cell"] = np.digitize(grid["x_in"], [-0.83, -0.28, 0.28, 0.83]) * 5 + \
        np.digitize(grid["z_norm"], [0.0, 0.33, 0.67, 1.0])

    # 타자 프로필 붙이기
    cell = prof["cell"][prof["cell"]["batter"] == batter_id].set_index("cell")["b_cell_rv"]
    grid["b_cell_rv"] = grid["cell"].map(cell).fillna(grid["cell"].map(prof["cell_league"]))
    ov = prof["overall"].set_index("batter")
    for c in ov.columns:                       # 타자 전체 성향 (타구속도, 발사각, 방향 등)
        v = ov[c].get(batter_id, np.nan)
        grid[c] = prof["defaults"][c] if pd.isna(v) else v
    if "loc" in prof:                          # 코스별 당겨치기
        grid["xloc"] = np.digitize(grid["x_in"], [-0.28, 0.28])
        loc = prof["loc"][prof["loc"]["batter"] == batter_id].set_index("xloc")["b_pull_loc"]
        grid["b_pull_loc"] = grid["xloc"].map(loc).fillna(
            grid["xloc"].map(prof["defaults"]["b_pull_loc"]))
    if "oz" in prof:                           # 코스 구역별 스윙률 (선구안)
        from build_matchup import zone_region
        grid["oz"] = zone_region(grid["x_in"], grid["z_norm"])
        oz = prof["oz"][prof["oz"]["batter"] == batter_id].set_index("oz")["b_swing_oz"]
        grid["b_swing_oz"] = grid["oz"].map(oz).fillna(grid["oz"].map(prof["defaults"]["b_swing_oz"]))
    grp_prof = prof["group"][prof["group"]["batter"] == batter_id].set_index("group")
    grp_cols = [c for c in prof["group"].columns if c not in ("batter", "group")]

    rows = []
    for _, p in ars.iterrows():
        g = grid.copy()
        for c in ["grp", "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
                  "rel_z", "rel_side", "ext", "vaa_rel",       # 구종 특성 + 릴리스·진입각
                  "height", "weight", "rel_z_h", "ext_h"]:     # + 신체조건
            if c in p:
                g[c] = p[c]
        for c in grp_cols:                     # 구종 계열별 타자 성향
            v = grp_prof[c].get(p["group"], np.nan) if c in grp_prof else np.nan
            g[c] = prof["defaults"][c][p["group"]] if pd.isna(v) else v
        g["balls"], g["strikes"] = balls, strikes
        g["same_hand"] = int(stand == p["p_throws"])
        g["outs"], g["r1"], g["r2"], g["r3"] = sit
        for k, v in (extra or {}).items():     # 추가 상황 정보 (예: 수비 배치)
            g[k] = v
        if prev is None:                      # 직전 공 모름
            for c in ["prev_grp", "prev_velo", "prev_x", "prev_z",
                      "d_velo", "d_x", "d_z", "d_pfx_x", "d_pfx_z"]:
                g[c] = np.nan
        else:
            g["prev_grp"], g["prev_velo"] = prev["grp"], prev["release_speed"]
            g["prev_x"], g["prev_z"] = prev["x"], prev["z"]
            g["d_velo"] = g["release_speed"] - prev["release_speed"]
            g["d_x"], g["d_z"] = g["x_in"] - prev["x"], g["z_norm"] - prev["z"]
            g["d_pfx_x"] = g["pfx_x"] - prev["pfx_x"]
            g["d_pfx_z"] = g["pfx_z"] - prev["pfx_z"]
        g["rv"] = M["model"].predict(g[M["features"]])
        cols = ["rv"]
        if S is not None:                     # 스윙/헛스윙 모델이 있으면 같이 계산
            g["p_swing"] = S["swing"].predict_proba(g[S["features"]])[:, 1]
            g["p_whiff"] = S["whiff"].predict_proba(g[S["features"]])[:, 1]
            if S.get("embed") is not None:    # 신경망+임베딩과 평균 (2025 검증: 헛스윙 AUC 0.787 → 0.797)
                ps, pw = embed_swing(S["embed"], g, pitcher_id, batter_id)
                g["p_swing"] = (g["p_swing"] + ps) / 2
                g["p_whiff"] = (g["p_whiff"] + pw) / 2
            cols += ["p_swing", "p_whiff"]
        res = g.groupby(["tx", "tz"])[cols].mean().reset_index()
        res["pitch_type"], res["usage"], res["velo"] = p["pitch_type"], p["usage"], p["release_speed"]
        rows.append(res)
    out = pd.concat(rows, ignore_index=True)
    out["rv100"] = out["rv"] * 100      # 100구 던졌을 때 실점 변화
    return out


def heatmap(res, title, path, stand, korean=False):
    """구종별 코스 히트맵. path가 있으면 파일로 저장, 항상 그림(fig)을 돌려줌 (앱에서 사용)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    types = res.groupby("pitch_type")["usage"].first().sort_values(ascending=False).index
    vmax = np.abs(res["rv100"]).quantile(0.95)
    fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 4.2), squeeze=False)
    # 투수 시점: 우타자는 오른쪽에 서 있음 → 오른쪽이 몸쪽.
    # 좌타자는 왼쪽에 서 있음 → 좌우를 뒤집어서 왼쪽이 몸쪽.
    flip = -1 if stand == "L" else 1
    bat_x = 1.22 * flip
    for ax, t in zip(axes[0], types):
        d = res[res["pitch_type"] == t].pivot(index="tz", columns="tx", values="rv100")
        if flip == -1:
            d = d.iloc[:, ::-1]
        im = ax.imshow(d.values, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax,
                       extent=[X_TARGETS[0], X_TARGETS[-1], Z_TARGETS[0], Z_TARGETS[-1]],
                       aspect="auto")
        ax.add_patch(Rectangle((ZONE_X[0], ZONE_Z[0]), ZONE_X[1] - ZONE_X[0],
                               ZONE_Z[1] - ZONE_Z[0], fill=False, lw=2, color="black"))
        best = res[res["pitch_type"] == t].nsmallest(1, "rv100").iloc[0]
        ax.plot(best["tx"] * flip, best["tz"], "k*", ms=14)
        bat = ("우타자" if stand == "R" else "좌타자") if korean else ("RHB" if stand == "R" else "LHB")
        ax.text(bat_x, 0.5, bat, rotation=90,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="black", bbox=dict(facecolor="white", alpha=0.8, lw=0))
        ax.set_title(f"{t}  ({best['rv100']:+.1f})", fontsize=12)
        ax.set_xticks([])
        if korean:
            ax.set_xlabel("몸쪽 ←   → 바깥쪽" if flip == -1 else "바깥쪽 ←   → 몸쪽")
        else:
            ax.set_xlabel("inside  <--   -->  away" if flip == -1 else "away  <--   -->  inside")
        ax.set_yticks([])
    if korean:
        axes[0][0].set_ylabel("낮게 ←   → 높게")
        fig.suptitle(title + "  (투수 시점)\n파랑 = 투수 유리, 빨강 = 타자 유리, ★ = 최적 코스",
                     fontsize=11, y=1.08)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="100구당 상대 득점")
    else:
        axes[0][0].set_ylabel("low  <--  -->  high")
        fig.suptitle(title + "  (pitcher's view)"
                     "\nblue = good for pitcher, red = good for batter, * = best spot",
                     fontsize=11, y=1.08)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="runs per 100 pitches")
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitcher", required=True)
    ap.add_argument("--batter", required=True)
    ap.add_argument("--count", default="0-0", help="볼-스트라이크, 예: 1-2")
    ap.add_argument("--stand", choices=["L", "R"], help="스위치 타자면 타석 방향 지정")
    ap.add_argument("--prev", help="직전 공 구종, 예: FF")
    ap.add_argument("--prev-spot", default="", help="직전 공 코스, 예: '높게 몸쪽', '낮게 떨어지게 바깥쪽'")
    ap.add_argument("--outs", type=int, default=0, choices=[0, 1, 2], help="아웃카운트")
    ap.add_argument("--runners", default="", help="주자 위치, 예: 1  /  1,3  /  만루  (비우면 주자 없음)")
    ap.add_argument("--command", type=float, default=0.5,
                    help="제구 오차(피트). 작을수록 제구 좋은 투수로 가정")
    a = ap.parse_args()

    with open("matchup_model.pkl", "rb") as f:
        M = pickle.load(f)
    pnames = dict(zip(M["arsenal"]["pitcher"], M["arsenal"]["player_name"]))
    pid, pname = find(pnames, a.pitcher, "투수")
    bid, bname = find(M["batter_names"], a.batter, "타자")
    if bid not in M["batter_zone"].index:
        sys.exit(f"{bname}는 이 기간 데이터가 없어요.")
    stand = a.stand or M["batter_stand"].get(bid, "R")
    balls, strikes = map(int, a.count.split("-"))

    prev = None
    if a.prev:
        ars = M["arsenal"][M["arsenal"]["pitcher"] == pid].set_index("pitch_type")
        if a.prev not in ars.index:
            sys.exit(f"{pname}의 구종에 {a.prev}가 없어요. 가능한 구종: {', '.join(ars.index)}")
        prev = ars.loc[a.prev].to_dict()
        prev["x"], prev["z"] = parse_spot(a.prev_spot)
    elif (balls, strikes) != (0, 0):
        print("※ 직전 공(--prev)을 안 넣어서 '직전 공 모름'으로 계산해요.")

    S = None
    if os.path.exists("swing_model.pkl"):
        with open("swing_model.pkl", "rb") as f:
            S = pickle.load(f)
    r = "1,2,3" if a.runners.strip() == "만루" else a.runners
    bases = {b.strip() for b in r.replace(" ", ",").split(",") if b.strip()}
    sit = (a.outs, int("1" in bases), int("2" in bases), int("3" in bases))
    res = simulate(M, pid, bid, balls, strikes, stand, a.command, prev, S, sit)

    print(f"\n{pname}  vs  {bname} ({'좌' if stand == 'L' else '우'}타)  |  카운트 {balls}-{strikes}")
    on = [f"{b}루" for b, v in zip("123", sit[1:]) if v]
    print(f"상황: {sit[0]}아웃, " + ("만루" if len(on) == 3 else ", ".join(on) + " 주자" if on else "주자 없음"))
    if prev is not None:
        print(f"직전 공: {a.prev} ({describe(prev['x'], prev['z'])})")
    print("숫자 = 100구당 실점 변화.  -면 투수에게 유리, +면 타자에게 유리\n")

    best = res.loc[res.groupby("pitch_type")["rv100"].idxmin()].sort_values("rv100")
    worst = res.loc[res.groupby("pitch_type")["rv100"].idxmax()].set_index("pitch_type")
    sw = S is not None
    head = f"{'순위':<4}{'구종':<5}{'구속':>6}  {'최적 코스':<16}{'기대값':>7}"
    if sw:
        head += f"{'스윙률':>7}{'헛스윙률':>8}"
    print(head + "   피해야 할 코스")
    for i, (_, r) in enumerate(best.iterrows(), 1):
        w = worst.loc[r["pitch_type"]]
        line = (f"{i:<5}{r['pitch_type']:<5}{r['velo']:>5.1f}  {describe(r['tx'], r['tz']):<16}"
                f"{r['rv100']:>+7.1f}")
        if sw:
            line += f"{r['p_swing']:>8.0%}{r['p_whiff']:>9.0%}"
        print(line + f"   {describe(w['tx'], w['tz'])} ({w['rv100']:+.1f})")
    if sw:
        print("(스윙률 = 타자가 배트를 낼 확률, 헛스윙률 = 배트를 냈을 때 헛칠 확률)")

    # 비교 기준: 평소 구사율대로 구종을 고르고, 존 근처 아무 데나 던졌을 때
    near = res[res["tx"].between(-1.1, 1.1) & res["tz"].between(-0.3, 1.2)]
    w = near.groupby("pitch_type").agg(rv=("rv100", "mean"), usage=("usage", "first"))
    base = (w["rv"] * w["usage"]).sum() / w["usage"].sum()

    top = best.iloc[0]
    print(f"\n▶ 추천: {top['pitch_type']}를 {describe(top['tx'], top['tz'])}")
    print(f"  아무 생각 없이 던질 때 {base:+.1f}  →  추천대로 던지면 {top['rv100']:+.1f}"
          f"  (100구당 {base - top['rv100']:.1f}점 절약)")

    tag = f"_after_{a.prev}" if prev is not None else ""
    path = f"pitch_plan_{pname.split(',')[0]}_vs_{bname.split(',')[0]}_{balls}-{strikes}{tag}.png"
    heatmap(res, f"{pname} vs {bname}  count {balls}-{strikes}", path, stand)
    print(f"코스별 히트맵 저장: {path}")
