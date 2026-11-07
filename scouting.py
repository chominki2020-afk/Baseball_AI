"""
시리즈 스카우팅 리포트 — 상대 타선 9명 × 우리 선발들에 대한 공략법을 인쇄용 문서 한 개로

검증 결과에 맞춰 무엇을 보여줄지 정했다
  - 코스 추천: 2025 백테스트로 검증됨 → 구종별 목표 코스를 그대로 보여 줌
  - 구종 '순위': 검증 안 됨 → "1순위 구종" 대신 게임이론 구종 비율(game.py, β = MLB 투수 실제 선택으로 추정)
  - 수비 위치: 타구 방향 분포(실제 기록) 중심, 모델 값은 참고
  - 스윙·헛스윙 확률: LightGBM + 신경망 임베딩 평균 (2025 검증에서 가장 정확)

실행: python scouting.py --opp NYY --us PIT                # 상대 = 양키스, 우리 = 피츠버그 선발 3명 (2~3분)
      python scouting.py --opp LAD --pitchers "Skenes" "Cole" "Sale"
      python scouting.py --teams                              # 팀별 라인업·선발 표 (원본 CSV 필요, 앱은 이 표만 사용)
"""
import argparse
import html
import os
import pickle
import time

import numpy as np
import pandas as pd

import game as GM
import recommend as R
import strategy as ST

DATA = "statcast_2024-04-01_2024-09-30.csv"      # 모델이 학습한 시즌 (타자 프로필·구종 정보가 이 시즌 기준)
PITCH_KO = {"FF": "포심", "SI": "싱커", "FC": "커터", "SL": "슬라이더", "ST": "스위퍼", "CU": "커브", "KC": "너클커브",
            "SV": "슬러브", "CS": "슬로커브", "CH": "체인지업", "FS": "스플리터", "FO": "포크", "SC": "스크루볼"}
COLOR = {"FF": "#d22d49", "SI": "#fe9d00", "FC": "#933f2c", "SL": "#c9c400", "ST": "#ddb33a", "CU": "#00b4d0",
         "KC": "#6236cd", "SV": "#7f9cc7", "CS": "#0068ff", "CH": "#1dbe3a", "FS": "#3bacac", "FO": "#55ccab",
         "SC": "#60db33"}
GROUP_KO = {"fastball": "패스트볼", "breaking": "변화구", "offspeed": "오프스피드"}
GROUP_TOPIC = {"fastball": "패스트볼은", "breaking": "변화구는", "offspeed": "오프스피드는"}
COUNTS = [((0, 0), "초구"), ((1, 2), "유리 (1-2)"), ((2, 0), "불리 (2-0)"), ((3, 2), "풀카운트")]


# ------------------------------------------------------------------ 데이터: 타선·선발 뽑기
def team_games(path=DATA):
    """타석마다 공격 팀·수비 팀 (Statcast 원본의 홈/원정 + 초/말)"""
    d = pd.read_csv(path, usecols=["game_pk", "at_bat_number", "batter", "pitcher", "inning", "inning_topbot",
                                   "home_team", "away_team"])
    top = d["inning_topbot"] == "Top"
    d["bat_team"] = np.where(top, d["away_team"], d["home_team"])
    d["fld_team"] = np.where(top, d["home_team"], d["away_team"])
    return d.drop_duplicates(["game_pk", "at_bat_number"])


def common_lineup(pa, team, M):
    """가장 자주 나온 선발 라인업: 경기마다 첫 9명 타순 → 타순 자리별로 가장 많이 선 타자 (중복 없이)"""
    t = pa[pa["bat_team"] == team].sort_values(["game_pk", "at_bat_number"])
    first = t.drop_duplicates(["game_pk", "batter"])
    first = first.assign(slot=first.groupby("game_pk").cumcount() + 1)
    first = first[first["slot"] <= 9]
    cnt = first.groupby(["slot", "batter"]).size().rename("n").reset_index()
    known = set(M["batter_zone"].index)
    picked, order = set(), []
    for s in range(1, 10):
        c = cnt[(cnt["slot"] == s) & (~cnt["batter"].isin(picked)) & (cnt["batter"].isin(known))]
        if c.empty:
            continue
        b = int(c.sort_values("n", ascending=False).iloc[0]["batter"])
        picked.add(b)
        order.append((s, b, int(first[first["batter"] == b]["game_pk"].nunique())))
    return order


def top_starters(pa, team, M, k=3):
    """그 팀에서 선발로 가장 많이 나온 투수 k명 (경기 첫 타석의 투수 = 선발)"""
    t = pa[pa["fld_team"] == team].sort_values(["game_pk", "at_bat_number"])
    st_ = t.drop_duplicates("game_pk")["pitcher"].value_counts()
    known = set(M["arsenal"]["pitcher"])
    return [int(p) for p in st_.index if p in known][:k]


TEAMS_FILE = "scouting_teams.pkl"


def team_table(M, path=DATA):
    """30개 팀의 대표 라인업·선발 3명 (원본 CSV 없이도 앱에서 리포트를 만들 수 있게 저장)"""
    pa = team_games(path)
    teams = sorted(set(pa["bat_team"]))
    return {t: {"lineup": common_lineup(pa, t, M), "starters": top_starters(pa, t, M)} for t in teams}


def load_teams(M):
    if os.path.exists(TEAMS_FILE):
        return pickle.load(open(TEAMS_FILE, "rb"))
    T = team_table(M)
    pickle.dump(T, open(TEAMS_FILE, "wb"))
    return T


# ------------------------------------------------------------------ 타자 성향
def batter_profile(M, bid):
    prof, d = M["profile"], M["profile"]["defaults"]
    ov = prof["overall"].set_index("batter")
    val = lambda c: d[c] if (c not in ov.columns or pd.isna(ov[c].get(bid, np.nan))) else ov[c][bid]
    gp = prof["group"][prof["group"]["batter"] == bid].set_index("group")
    grp = {}
    for g in ("fastball", "breaking", "offspeed"):
        gv = lambda c: d[c][g] if (c not in gp.columns or pd.isna(gp[c].get(g, np.nan))) else gp[c][g]
        grp[g] = {"whiff": gv("b_whiff"), "chase": gv("b_chase"), "xwoba": gv("b_xwoba"),
                  "lg_whiff": d["b_whiff"][g], "lg_chase": d["b_chase"][g], "lg_xwoba": d["b_xwoba"][g]}
    cell = prof["cell"][prof["cell"]["batter"] == bid].set_index("cell")["b_cell_rv"]
    grid = np.full((5, 5), np.nan)                    # [높이, 좌우] 투수 시점, 바깥쪽(-) → 몸쪽(+) (우타 기준)
    for c in range(25):
        grid[c % 5, c // 5] = cell.get(c, prof["cell_league"].get(c, np.nan)) * 100
    return {"ev": val("b_ev"), "la": val("b_la"), "pull": val("b_pull"), "oppo": val("b_oppo"),
            "lg_ev": d["b_ev"], "lg_la": d["b_la"], "lg_pull": d["b_pull"], "group": grp, "grid": grid}


def archetype(BS, bid):
    if BS is None or bid not in BS["report"].index:
        return None, None
    p = float(BS["report"].loc[bid, "bat_speed_pct"])
    return ("컨택형" if p < 33 else "파워형" if p >= 67 else "균형형"), p


def defense_tips(M, Bm, bid, stand):
    """타구 방향 분포(실제 기록)로 수비 위치 — 앱 '추천 수비 위치'와 같은 규칙"""
    if Bm is None:
        return [], None
    bip, fallback = ST.batter_bip(Bm, bid, stand)
    lg = ST.spray_summary(Bm["batter_bip"][Bm["batter_bip"]["stand"] == stand])
    s = ST.spray_summary(bip)
    gb, lgb = s["땅볼"]["당김"], lg["땅볼"]["당김"]
    air, lair = s["공중 타구"]["당김"], lg["공중 타구"]["당김"]
    pull_side = "3루" if stand == "R" else "1루"
    tips = []
    if gb >= lgb + 0.05:
        tips.append(f"<b>내야</b> 땅볼 {gb:.0%} 당김 (리그 {lgb:.0%}) → <b>{pull_side} 쪽 쉐이드</b>")
    elif gb <= lgb - 0.05:
        tips.append(f"<b>내야</b> 땅볼 당김 {gb:.0%} (리그 {lgb:.0%}) → <b>정상, 반대쪽 간격 좁히기</b>")
    else:
        tips.append(f"<b>내야</b> 땅볼 방향 리그 평균 수준 ({gb:.0%} 당김) → <b>정상 수비</b>")
    if air >= lair + 0.05:
        tips.append(f"<b>외야</b> 공중 타구 {air:.0%} 당김 (리그 {lair:.0%}) → "
                    f"<b>{'좌익·중견' if stand == 'R' else '우익·중견'} 당긴 쪽으로</b>")
    else:
        tips.append(f"<b>외야</b> 공중 타구 방향 고름 ({air:.0%} 당김) → <b>정상</b>")
    loc = M["profile"]["loc"]
    lp = loc[loc["batter"] == bid].set_index("xloc")["b_pull_loc"]
    dl = M["profile"]["defaults"]["b_pull_loc"]
    pin, pout = lp.get(2, dl[2]), lp.get(0, dl[0])
    tips.append(f"<b>투구와 연결</b> 몸쪽 공 {pin:.0%} 당김 / 바깥쪽 {pout:.0%} → 당긴 쪽 수비면 몸쪽, 정상 수비면 바깥쪽으로")
    return tips, fallback


def similar(EMB, M, bid, k=3):
    if EMB is None:
        return []
    import embed as EM
    s = EM.similar(EMB, "batter", bid, k=k, min_n=500)
    if s is None:
        return []
    return [M["batter_names"].get(int(i), str(i)) for i in s["id"]]


# ------------------------------------------------------------------ 공략법 (타자 × 투수 × 카운트)
def plan(M, S, pid, bid, stand, balls, strikes, beta):
    res = R.simulate(M, pid, bid, balls, strikes, stand, 0.5, None, S, (0, 0, 0, 0), n_noise=25)
    df, ev = GM.plan(res, beta)
    rows = []
    for _, r in df.sort_values("mix", ascending=False).iterrows():
        g = res[res["pitch_type"] == r["pitch_type"]]
        best = g.loc[g["rv100"].idxmin()]
        # 한가운데를 뺀 가장 좋은 곳 (목표가 한가운데일 때 '가장자리라면 어디, 몇 점 손해'를 같이 보여 주려고)
        edge = g[~((g["tx"].abs() <= 0.28) & g["tz"].between(0.33, 0.67))]
        alt = edge.loc[edge["rv100"].idxmin()]
        rows.append({"pitch_type": r["pitch_type"], "mix": r["mix"], "usage": r["usage"], "rv0": r["rv0"],
                     "tx": best["tx"], "tz": best["tz"], "where": R.describe(best["tx"], best["tz"]),
                     "alt_tx": alt["tx"], "alt_tz": alt["tz"], "alt_where": R.describe(alt["tx"], alt["tz"]),
                     "alt_cost": alt["rv100"] - best["rv100"],
                     "p_whiff": best.get("p_whiff", np.nan), "p_swing": best.get("p_swing", np.nan)})
    return pd.DataFrame(rows), ev


def build(M, S, Bm, BS, EMB, opp, us, lineup, starters, beta=GM.DEFAULT_BETA, log=print):
    pnames = dict(zip(M["arsenal"]["pitcher"], M["arsenal"]["player_name"]))
    out = {"opp": opp, "us": us, "beta": beta, "batters": [], "pitchers": [],
           "made": time.strftime("%Y-%m-%d")}
    for pid in starters:
        a = M["arsenal"][M["arsenal"]["pitcher"] == pid].sort_values("usage", ascending=False)
        out["pitchers"].append({"id": pid, "name": pnames[pid], "hand": a["p_throws"].iloc[0],
                                "arsenal": [(t, u, v) for t, u, v in zip(a["pitch_type"], a["usage"], a["release_speed"])
                                            if u >= 0.03]})
    t0 = time.time()
    for slot, bid, games in lineup:
        name = M["batter_names"].get(bid, str(bid))
        stand_mode = M["batter_stand"].get(bid, "R")
        b = {"slot": slot, "id": bid, "name": name, "games": games, "stand": stand_mode,
             "profile": batter_profile(M, bid), "similar": similar(EMB, M, bid)}
        b["arch"], b["bs_pct"] = archetype(BS, bid)
        b["tips"], b["tips_fallback"] = defense_tips(M, Bm, bid, stand_mode if stand_mode in "RL" else "R")
        b["plans"] = {}
        for p in out["pitchers"]:
            stand = stand_mode if stand_mode in ("R", "L") else ("L" if p["hand"] == "R" else "R")
            for (bl, sk), _ in COUNTS:
                b["plans"][(p["id"], bl, sk)] = plan(M, S, p["id"], bid, stand, bl, sk, beta)
        out["batters"].append(b)
        log(f"  {slot}번 {name} 완료 ({time.time() - t0:.0f}초)")
    return out


# ------------------------------------------------------------------ 한 줄 요약 (근거 숫자와 함께)
def key_points(b):
    p, pts = b["profile"], []
    grp = p["group"]
    weak = max(grp, key=lambda g: grp[g]["whiff"] - grp[g]["lg_whiff"])
    w = grp[weak]
    if w["whiff"] - w["lg_whiff"] >= 0.03:
        pts.append(f"{GROUP_KO[weak]}에 헛스윙 {w['whiff']:.0%} (리그 {w['lg_whiff']:.0%})")
    chase = max(grp, key=lambda g: grp[g]["chase"] - grp[g]["lg_chase"])
    c = grp[chase]
    if c["chase"] - c["lg_chase"] >= 0.03:
        pts.append(f"존 밖 {GROUP_KO[chase]}에 잘 따라 나옴 ({c['chase']:.0%}, 리그 {c['lg_chase']:.0%})")
    strong = max(grp, key=lambda g: grp[g]["xwoba"] - grp[g]["lg_xwoba"])
    s = grp[strong]
    if s["xwoba"] - s["lg_xwoba"] >= 0.02:
        pts.append(f"{GROUP_TOPIC[strong]} 맞으면 위험 (타구 xwOBA {s['xwoba']:.3f}, 리그 {s['lg_xwoba']:.3f})")
    g = p["grid"][1:4, 1:4]                            # 스트라이크존 안 3×3
    hi = tuple(int(v) for v in np.unravel_index(np.nanargmax(g), g.shape))
    lo = tuple(int(v) for v in np.unravel_index(np.nanargmin(g), g.shape))
    nm = lambda ij: "한가운데" if ij == (1, 1) else f"{['낮은', '가운데 높이', '높은'][ij[0]]} {['바깥쪽', '가운데', '몸쪽'][ij[1]]}"
    pts.append(f"존 안에서 타자가 가장 강한 곳: {nm(hi)} ({g[hi]:+.1f}) · 가장 약한 곳: {nm(lo)} ({g[lo]:+.1f})")
    return pts


# ------------------------------------------------------------------ HTML
CSS = """
@page { size: A4; margin: 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Noto Sans KR','Noto Sans CJK KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif; color:#1f2328;
       margin:0; background:#fff; font-size:12px; line-height:1.45; }
.page { max-width: 186mm; margin: 0 auto; padding: 6mm 0; page-break-after: always; }
.page:last-child { page-break-after: auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 0 0 6px; } h3 { font-size: 13px; margin: 10px 0 4px; }
.sub { color:#59636e; font-size: 11px; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background:#f6f8fa; font-weight:600; font-size: 11px; }
.num { text-align:right; font-variant-numeric: tabular-nums; }
.tag { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; background:#eef2ff; color:#3730a3; margin-left:6px; }
.box { border:1px solid #e5e7eb; border-radius:8px; padding:8px 10px; margin-top:8px; }
.grid2 { display:grid; grid-template-columns: 160px 1fr; gap: 12px; align-items:start; }
.pitch { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; vertical-align:middle; }
.bar { display:inline-block; height:9px; border-radius:2px; vertical-align:middle; margin-right:4px; }
ul { margin: 4px 0 0 16px; padding:0; } li { margin: 2px 0; }
.foot { color:#59636e; font-size:10px; margin-top:10px; border-top:1px solid #e5e7eb; padding-top:6px; }
.plan td { font-size: 11px; }
.mix { color:#59636e; font-size:10.5px; line-height:1.6; }
.tg { margin-top:2px; }
.zone td { width:27px; height:24px; text-align:center; font-size:9px; border:1px solid #fff; padding:0; }
.zone .in { outline: 2px solid #1f2328; outline-offset:-2px; }
"""


def esc(x):
    return html.escape(str(x))


def flipname(n):
    """'Judge, Aaron' → 'Aaron Judge'"""
    return " ".join(reversed([s.strip() for s in str(n).split(",")])) if "," in str(n) else str(n)


def zone_html(grid, stand):
    g = grid[:, ::-1] if stand == "L" else grid      # 좌타자: 왼쪽이 몸쪽
    vmax = np.nanmax(np.abs(g)) or 1
    rows = []
    for i in range(4, -1, -1):                        # 위 = 높게
        tds = []
        for j in range(5):
            v = g[i, j]
            a = min(abs(v) / vmax, 1) * 0.85
            col = f"rgba(214,39,40,{a:.2f})" if v > 0 else f"rgba(31,119,180,{a:.2f})"
            cls = ' class="in"' if 1 <= i <= 3 and 1 <= j <= 3 else ""
            tds.append(f'<td{cls} style="background:{col}">{v:+.1f}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    side = "몸쪽 ← → 바깥쪽" if stand == "L" else "바깥쪽 ← → 몸쪽"
    return (f'<table class="zone" style="width:auto">{"".join(rows)}</table>'
            f'<div class="sub">투수 시점 · {side}<br>빨강 = 타자가 강한 코스 (100구당 실점)</div>')


SIDE_S = {"바깥쪽 빠지게": "바깥 볼", "바깥쪽": "바깥", "가운데": "가운데", "몸쪽": "몸쪽", "몸쪽 빠지게": "몸쪽 볼"}
HEIGHT_S = {"낮게 떨어지게": "낮게 볼", "낮게": "낮게", "가운데 높이": "중간", "높게": "높게", "높게 빠지게": "높게 볼"}


TIE = 0.3      # 100구당 실점 차이가 이보다 작으면 같은 것으로 봄 (백테스트 구간 폭 ±1.4보다 훨씬 작음)


def heart(r):
    return abs(r["tx"]) <= 0.28 and 0.33 <= r["tz"] <= 0.67


def short(where):
    side, height = [x.strip() for x in where.split(",")]
    return f"{SIDE_S.get(side, side)}·{HEIGHT_S.get(height, height)}"


def cell_html(df):
    """한 칸 = 구종 비율 + 비율 상위 두 구종의 목표 코스"""
    chips = []
    for _, r in df[df["mix"] >= 0.10].iterrows():
        c = COLOR.get(r["pitch_type"], "#888")
        chips.append(f'<span class="bar" style="width:{max(r["mix"] * 40, 3):.0f}px;background:{c}"></span>'
                     f'{esc(PITCH_KO.get(r["pitch_type"], r["pitch_type"]))} {r["mix"] * 100:.0f}')
    tg = []
    for _, r in df.head(2).iterrows():
        dot = (f'<span class="pitch" style="background:{COLOR.get(r["pitch_type"], "#888")}"></span>'
               f'{esc(PITCH_KO.get(r["pitch_type"], r["pitch_type"]))} ')
        if heart(r) and r["alt_cost"] < TIE:     # 한가운데와 차이가 거의 없으면 가장자리를 목표로
            tg.append(dot + f'<b>{esc(short(r["alt_where"]))}</b>≈')
        elif heart(r):                           # 존 안 승부 + 가장자리 대안과 그 비용
            tg.append(dot + f'<b>존 안 승부</b>※ <span class="sub">가장자리면</span> <b>{esc(short(r["alt_where"]))}</b>'
                            f' <span class="sub">+{r["alt_cost"]:.1f}</span>')
        else:
            wh = f' <span class="sub">헛{r["p_whiff"] * 100:.0f}%</span>' if pd.notna(r["p_whiff"]) else ""
            tg.append(dot + f'<b>{esc(short(r["where"]))}</b>{wh}')
    return f'<div class="mix">{" ".join(chips)}</div><div class="tg">{"<br>".join(tg)}</div>'


def render(rep):
    opp, us = rep["opp"], rep["us"]
    pages = []
    # --- 표지: 시리즈 요약
    lineup_rows = []
    for b in rep["batters"]:
        kp = key_points(b)
        arch = f'<span class="tag">{esc(b["arch"])}</span>' if b["arch"] else ""
        lineup_rows.append(f'<tr><td class="num">{b["slot"]}</td><td><b>{esc(flipname(b["name"]))}</b>{arch}</td>'
                           f'<td>{"좌" if b["stand"] == "L" else "스위치" if b["stand"] == "S" else "우"}</td>'
                           f'<td>{esc(kp[0]) if kp else ""}</td></tr>')
    pit_rows = []
    for i, p in enumerate(rep["pitchers"], 1):
        ars = " ".join(f'<span class="pitch" style="background:{COLOR.get(t, "#888")}"></span>{esc(PITCH_KO.get(t, t))} '
                       f'{u:.0%} ({v:.0f})' for t, u, v in p["arsenal"])
        pit_rows.append(f'<tr><td>{i}차전</td><td><b>{esc(flipname(p["name"]))}</b> ({"좌" if p["hand"] == "L" else "우"}투)</td>'
                        f'<td>{ars}</td></tr>')
    pages.append(f"""
<div class="page">
  <div class="sub">시리즈 스카우팅 리포트 · {esc(rep['made'])} 생성</div>
  <h1>{esc(us)} 투수진 vs {esc(opp)} 타선</h1>
  <div class="sub">상대 타선 = 2024 시즌 가장 자주 나온 선발 라인업 · 우리 선발 = 2024 선발 등판 상위 {len(rep['pitchers'])}명</div>
  <h3>우리 선발</h3>
  <table><tr><th>경기</th><th>투수</th><th>구종 (구사율, 평균 구속 mph)</th></tr>{''.join(pit_rows)}</table>
  <h3>상대 타선 한눈에</h3>
  <table><tr><th class="num">타순</th><th>타자</th><th>타석</th><th>가장 두드러진 점</th></tr>{''.join(lineup_rows)}</table>
  <div class="box"><b>읽는 법</b>
    <ul>
      <li><b>구종 비율</b>은 "제일 좋은 공 하나"가 아니라 게임이론으로 계산한 비율이다. 같은 공만 던지면 타자가 노리고 들어오기 때문이다.
          노림 강도(β = {rep['beta']:g})는 MLB 투수들의 실제 카운트별 선택에서 추정했다.</li>
      <li><b>목표 코스</b>는 구종별로 기대 실점이 가장 낮은 곳이다 (제구 오차 0.5피트 가정). 2025 백테스트에서 추천 코스 근처로 간 공은
          100구당 2.6점 덜 내줬다.</li>
      <li><b>헛스윙 %</b>는 그 코스에서 배트를 냈을 때 헛칠 확률이다 (LightGBM + 선수 임베딩 신경망 평균).</li>
    </ul>
  </div>
  <div class="foot">데이터: MLB Statcast 2024 (Baseball Savant). 모델은 2024로 학습하고 2025로 검증했다. 구종 '순위'는 검증되지 않아 비율로만 제시한다.
  개별 타자 예측의 불확실성은 크다 (검증 상관 0.3~0.5). 코치의 판단을 돕는 참고 자료로 쓴다.</div>
</div>""")
    # --- 타자별 페이지
    for b in rep["batters"]:
        p = b["profile"]
        arch = (f'<span class="tag">{esc(b["arch"])} · 배트 스피드 상위 {100 - b["bs_pct"]:.0f}%</span>'
                if b["arch"] else "")
        grp_rows = "".join(
            f'<tr><td>{GROUP_KO[g]}</td><td class="num">{v["whiff"]:.0%} <span class="sub">({v["lg_whiff"]:.0%})</span></td>'
            f'<td class="num">{v["chase"]:.0%} <span class="sub">({v["lg_chase"]:.0%})</span></td>'
            f'<td class="num">{v["xwoba"]:.3f} <span class="sub">({v["lg_xwoba"]:.3f})</span></td></tr>'
            for g, v in p["group"].items())
        heads = "".join(f'<th>{esc(flipname(pi["name"]))} <span class="sub">{"좌" if pi["hand"] == "L" else "우"}투</span></th>'
                        for pi in rep["pitchers"])
        plan_rows, n_heart = [], 0
        for (bl, sk), label in COUNTS:
            tds = ""
            for pi in rep["pitchers"]:
                df, _ = b["plans"][(pi["id"], bl, sk)]
                n_heart += sum(heart(r) for _, r in df.head(2).iterrows())
                tds += f"<td>{cell_html(df)}</td>"
            plan_rows.append(f"<tr><td><b>{label}</b></td>{tds}</tr>")
        chase = np.mean([v["chase"] for v in p["group"].values()])
        heart_note = (f"<div class='sub'>한가운데 쪽 목표 처리: 제구 오차(0.5피트)를 생각하면 존 끝을 노리다 볼이 되는 손해가 커서 모델이 한가운데 쪽을 고르기도 한다"
                      f" (이 타자 존 밖 스윙 {chase:.0%}). 가장자리와 차이가 100구당 {TIE}점 미만이면 가장자리를 목표로 적었고(≈),"
                      f" 그 이상이면 '※ 존 안 승부'와 함께 가장자리를 노릴 때의 대가를 적었다.</div>") if n_heart else ""
        sim = (" · 비슷한 타자(임베딩): " + ", ".join(esc(flipname(s)) for s in b["similar"])) if b["similar"] else ""
        tips = "".join(f"<li>{t}</li>" for t in b["tips"])
        fb = " <span class='sub'>(타구 표본 부족 → 같은 타석 방향 리그 평균)</span>" if b["tips_fallback"] else ""
        kp = "".join(f"<li>{esc(t)}</li>" for t in key_points(b))
        pages.append(f"""
<div class="page">
  <div class="sub">{esc(opp)} {b['slot']}번 타자 · {b['games']}경기 선발 (2024)</div>
  <h2>{esc(flipname(b['name']))} <span class="sub">{"좌타" if b["stand"] == "L" else "스위치" if b["stand"] == "S" else "우타"}</span>{arch}</h2>
  <div class="sub">평균 타구속도 {p['ev']:.1f} mph (리그 {p['lg_ev']:.1f}) · 발사각 {p['la']:.1f}° (리그 {p['lg_la']:.1f}) ·
    당겨친 타구 {p['pull']:.0%} (리그 {p['lg_pull']:.0%}){sim}</div>
  <div class="grid2" style="margin-top:8px">
    <div>{zone_html(p['grid'], b['stand'] if b['stand'] in 'RL' else 'R')}</div>
    <div>
      <b>핵심</b><ul>{kp}</ul>
      <table style="margin-top:6px"><tr><th>구종 계열</th><th class="num">헛스윙 (리그)</th><th class="num">존 밖 스윙 (리그)</th>
        <th class="num">타구 xwOBA (리그)</th></tr>{grp_rows}</table>
    </div>
  </div>
  <h3>공략법 — 카운트 × 투수 <span class="sub">위: 구종 비율 (게임이론, %) · 아래: 비율 상위 두 구종의 목표 코스 · 헛 = 배트를 냈을 때 헛스윙</span></h3>
  <table class="plan"><tr><th style="width:11%">카운트</th>{heads}</tr>{''.join(plan_rows)}</table>
  {heart_note}
  <h3>수비 위치{fb}</h3>
  <ul>{tips}</ul>
</div>""")
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>스카우팅 리포트 {esc(us)} vs {esc(opp)}</title><style>{CSS}</style></head><body>{''.join(pages)}</body></html>")


# ------------------------------------------------------------------ 실행
def load_models():
    M = pickle.load(open("matchup_model.pkl", "rb"))
    S = pickle.load(open("swing_model.pkl", "rb"))
    S = dict(S, embed=R.load_embed("embed_model.npz"))
    Bm = pickle.load(open("bip_model.pkl", "rb")) if os.path.exists("bip_model.pkl") else None
    try:
        import batter_mech as BM
        BS = BM.load("batter_swing.pkl")
    except Exception:
        BS = None
    try:
        import embed as EM
        EMB = EM.load("embed_model.npz")
    except Exception:
        EMB = None
    return M, S, Bm, BS, EMB


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", help="상대 팀 약자 (예: NYY, LAD)")
    ap.add_argument("--teams", action="store_true", help="팀별 라인업·선발 표만 다시 만들기 (scouting_teams.pkl)")
    ap.add_argument("--us", default="", help="우리 팀 약자 → 그 팀 선발 상위 3명 자동 선택")
    ap.add_argument("--pitchers", nargs="*", default=[], help="선발 투수 이름 (성으로 검색)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.teams:
        T = team_table(pickle.load(open("matchup_model.pkl", "rb")))
        pickle.dump(T, open(TEAMS_FILE, "wb"))
        raise SystemExit(f"저장: {TEAMS_FILE} ({len(T)}개 팀)")
    if not a.opp:
        ap.error("--opp 가 필요해요 (예: --opp NYY --us PIT)")
    M, S, Bm, BS, EMB = load_models()
    print("팀별 라인업·선발 불러오는 중...")
    T = load_teams(M)
    if a.opp not in T:
        raise SystemExit(f"팀 약자를 확인하세요: {', '.join(T)}")
    lineup = T[a.opp]["lineup"]
    if a.pitchers:
        pn = dict(zip(M["arsenal"]["pitcher"], M["arsenal"]["player_name"]))
        starters = [R.find(pn, q, "투수")[0] for q in a.pitchers]
        us = a.us or "우리 팀"
    else:
        starters, us = T[a.us]["starters"], a.us
    print(f"상대 {a.opp} 타선 {len(lineup)}명, 우리 선발 {len(starters)}명 → 계산 시작")
    rep = build(M, S, Bm, BS, EMB, a.opp, us, lineup, starters)
    out = a.out or f"scouting_{us}_vs_{a.opp}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(rep))
    print(f"저장: {out}")
