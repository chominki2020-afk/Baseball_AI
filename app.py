"""
야구 AI 전략 시스템 — Streamlit 웹앱

  🛡️ 수비팀: 볼배합 추천 + 이 타자 상대 수비 위치
  ⚔️ 공격팀: 상대 수비 배치를 넣으면 노릴 공 / 참을 공 + 타구 방향 전략
  🎯 투구 궤적: 구종·코스를 고르면 3D 궤적과 터널링(다른 공과 언제까지 똑같이 보이나)
  🤖 AI 분석관: 자연어로 질문하면 Claude가 위 모델들을 도구로 불러서 숫자 기반으로 답함 (Anthropic API 키 필요)
  ♟️ 게임이론 볼배합: 타자가 노리는 효과를 넣은 카운트별 최적 구종 비율(혼합 전략) + 미니맥스 검증
  🧬 선수 지도: 신경망이 배운 선수 임베딩으로 비슷한 선수 찾기·선수 지도
  🏏 타자 개선: 배트 트래킹(배트 스피드·어택 앵글·컨택 포인트 등)을 이렇게 바꾸면 헛스윙·타구 질이 어떻게 변하나
  🛠️ 투수 개선: 투수의 장점·제일 좋은 공 기준으로 릴리스(타점·옆 위치·익스텐션)를 바꾸면 헛스윙이 늘지

실행:  run_app.bat 더블클릭   (또는  pip install streamlit plotly  →  streamlit run app.py)

필요한 파일 (같은 폴더):
  matchup_model.pkl, swing_model.pkl   (build_matchup.py, build_swing.py)
  offense_model.pkl                    (build_offense.py)
  bip_model.pkl                        (build_bip.py)
  physics.pkl                          (build_physics.py)
  pitcher_report.pkl, pitch_samples.pkl (build_mechanics_data.py, 투수 개선 모드)
  recommend.py, strategy.py, trajectory.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd
import streamlit as st

import importlib

import recommend as R
import strategy as ST
import trajectory as T
import animation as AN
import mechanics as MC
import batter_mech as BM
import analyst as AI
import embed as EM
import game as GM

# 앱이 켜져 있는 동안 보조 파일(recommend/strategy/trajectory.py)을 고쳐도 새로고침하면 바로 반영되게
for _m in (R, ST, T, AN, MC, BM, AI, EM, GM):
    importlib.reload(_m)

# 그래프 한글 폰트 (윈도우: 맑은 고딕, 맥: 애플고딕)
_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:   # JP 글꼴도 한글 포함
    if _f in _fonts:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="야구 AI 전략 시스템", page_icon="⚾", layout="wide")

# 새 환경(Streamlit Cloud 등)에서 모델 파일이 없으면 GitHub Release에서 한 번 받아 옴
import bootstrap
if bootstrap.missing():
    with st.spinner("처음 실행이라 학습된 모델 파일(약 30MB)을 받는 중이에요... (1분 이내)"):
        bootstrap.ensure_models(log=lambda *a: None)

PITCH_KO = {"FF": "포심", "SI": "싱커", "FC": "커터", "SL": "슬라이더", "ST": "스위퍼",
            "CU": "커브", "KC": "너클커브", "SV": "슬러브", "CS": "슬로커브",
            "CH": "체인지업", "FS": "스플리터", "FO": "포크", "SC": "스크루볼"}
COLOR = {"FF": "#d22d49", "SI": "#fe9d00", "FC": "#933f2c", "SL": "#c9c400", "ST": "#ddb33a",
         "CU": "#00b4d0", "KC": "#6236cd", "SV": "#7f9cc7", "CS": "#0068ff",
         "CH": "#1dbe3a", "FS": "#3bacac", "FO": "#55ccab", "SC": "#60db33"}
GROUP_KO = {"fastball": "패스트볼", "breaking": "변화구", "offspeed": "오프스피드"}
IF_KO = {0: "정상 수비", 1: "내야 쉐이드", 2: "전략적 배치"}
OF_KO = {0: "외야 정상", 1: "외야 전략적 배치"}
SIDE = {"바깥쪽 빠지게": -1.05, "바깥쪽": -0.55, "가운데": 0.0, "몸쪽": 0.55, "몸쪽 빠지게": 1.05}
HEIGHT = {"높게 빠지게": 1.2, "높게": 0.85, "가운데": 0.5, "낮게": 0.15, "낮게 떨어지게": -0.2}


def ko(t):
    return f"{t} ({PITCH_KO.get(t, t)})"


def name_ko(t):
    return PITCH_KO.get(t, t)


# ---------------------------------------------------------------- 모델 불러오기
def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_all():
    return {"M": _load("matchup_model.pkl"), "S": _load("swing_model.pkl"),
            "O": _load("offense_model.pkl"), "B": _load("bip_model.pkl"),
            "P": _load("physics.pkl"), "t": os.path.getmtime("matchup_model.pkl"),
            "bio": pd.read_csv("player_bio.csv").set_index("pitcher") if os.path.exists("player_bio.csv") else None}


D = load_all()
M, S = D["M"], D["S"]
if S is not None:                          # 스윙·헛스윙 = LightGBM과 신경망+임베딩의 평균 (2025 검증에서 가장 좋았던 조합)
    S = dict(S, embed=R.load_embed("embed_model.npz"))
ars_all = M["arsenal"]
pitchers = (ars_all.groupby(["pitcher", "player_name"])["n"].sum()
            .reset_index().sort_values("player_name"))
pitcher_names = dict(zip(pitchers["player_name"], pitchers["pitcher"]))
batters = {v: k for k, v in M["batter_names"].items() if k in M["batter_zone"].index}
batter_list = sorted(batters)


@st.cache_data(show_spinner=False)
def run_defense(pid, bid, balls, strikes, stand, command, prev_key, sit, _t):
    return R.simulate(M, pid, bid, balls, strikes, stand, command, _prev(pid, prev_key), S, sit)


@st.cache_data(show_spinner=False)
def run_offense(pid, bid, balls, strikes, stand, prev_key, sit, align, _t):
    O = D["O"]
    extra = {"if_al": align[0], "of_al": align[1]}
    prev = _prev(pid, prev_key)
    out = []
    for key in ("swing_rv", "take_rv"):
        m = dict(M, model=O[key], features=O["features"])
        r = R.simulate(m, pid, bid, balls, strikes, stand, 0.15, prev, None, sit, n_noise=12, extra=extra)
        out.append(r.set_index(["pitch_type", "tx", "tz"])["rv100"])
    res = out[0].rename("swing").to_frame().join(out[1].rename("take")).reset_index()
    res["value"] = res["swing"] - res["take"]          # +면 치는 게 이득 (타자 입장)
    u = ars_all[ars_all["pitcher"] == pid].set_index("pitch_type")
    res["usage"] = res["pitch_type"].map(u["usage"])
    res["velo"] = res["pitch_type"].map(u["release_speed"])
    return res


@st.cache_resource
def load_motion_lib():
    return AN.load_library("obp_motions.npz")


@st.cache_data(show_spinner=False)
def whatif_swing(pid, bid, stand, t, d_h, d_s, d_e, dv, tx, tz, _t):
    """릴리스를 바꿨을 때 스윙·헛스윙 확률 (0-0, 그 코스에 정확히 들어갔다고 가정)"""
    a = ars_all[(ars_all["pitcher"] == pid) & (ars_all["pitch_type"] == t)].copy()
    a["rel_z"] += d_h
    a["rel_side"] += d_s
    a["ext"] += d_e
    a["vaa_rel"] += dv
    r = R.simulate(dict(M, arsenal=a), pid, bid, 0, 0, stand, 0.05, None, S, (0, 0, 0, 0), n_noise=6)
    i = ((r["tx"] - tx) ** 2 + (r["tz"] - tz) ** 2).idxmin()
    return float(r.loc[i, "p_swing"]), float(r.loc[i, "p_whiff"])


def _prev(pid, prev_key):
    if not prev_key:
        return None
    t, x, z = prev_key
    p = ars_all[(ars_all["pitcher"] == pid) & (ars_all["pitch_type"] == t)].iloc[0].to_dict()
    p["x"], p["z"] = x, z
    return p


# ---------------------------------------------------------------- 공통 그림/카드
def verdict(v):
    """100구당 상대 득점 → 한눈에 보는 유불리"""
    if v <= -3:
        return "🔵 투수 크게 유리"
    if v <= -1:
        return "🔵 투수 유리"
    if v < 1:
        return "⚪ 팽팽"
    if v < 3:
        return "🔴 타자 유리"
    return "🔴 타자 크게 유리"


def rel_label(v, base):
    """평소(이 타자에게 평소 구사율대로 던질 때)와 비교한 추천 정도"""
    d = v - base
    if d <= -3:
        return "✅ 강력 추천"
    if d <= -1:
        return "👍 추천"
    if d < 1:
        return "➖ 평소 수준"
    return "⚠️ 비추천"


def in_zone(tx, tz):
    return -0.83 <= tx <= 0.83 and 0 <= tz <= 1


def reason(r):
    """왜 이 코스인지 한 줄 설명"""
    if in_zone(r["tx"], r["tz"]):
        if r["tz"] >= 0.67 and r.get("p_whiff", 0) >= 0.3:
            return "존 위쪽 · 헛스윙·빗맞은 뜬공 노림"
        return "존 안 · 스트라이크 잡고 약한 타구 유도"
    if r.get("p_whiff", 0) >= 0.5:
        return f"유인구 · 휘두르면 {r['p_whiff']:.0%} 헛스윙"
    return "존 경계 · 볼 되더라도 맞으면 약한 타구"


def scoreboard(balls, strikes, outs, r1, r2, r3):
    """중계 화면 같은 상황판 (다이아몬드 주자 + B/S/O 램프)"""
    def base(x, y, on):
        fill = "#f5a623" if on else "none"
        return (f'<rect x="{x - 11}" y="{y - 11}" width="22" height="22" transform="rotate(45 {x} {y})" '
                f'fill="{fill}" stroke="#9aa0a6" stroke-width="2.5"/>')

    def lamps(label, n, total, color, y):
        out = f'<text x="118" y="{y + 5}" font-size="15" font-weight="700" fill="#9aa0a6">{label}</text>'
        for i in range(total):
            fill = color if i < n else "none"
            out += f'<circle cx="{142 + i * 22}" cy="{y}" r="7.5" fill="{fill}" stroke="#9aa0a6" stroke-width="2"/>'
        return out
    return ('<svg viewBox="0 0 215 96" width="215" height="96" xmlns="http://www.w3.org/2000/svg">'
            + base(52, 22, r2) + base(84, 52, r1) + base(20, 52, r3)
            + lamps("B", balls, 3, "#34a853", 18) + lamps("S", strikes, 2, "#fbbc04", 46)
            + lamps("O", outs, 2, "#ea4335", 74) + "</svg>")


def zone_axes(ax, stand, zh, title):
    for x in (-0.28, 0.28):
        ax.plot([x, x], [0, 1], color="#c8ccd2", lw=1)
    for z in (1 / 3, 2 / 3):
        ax.plot([-0.83, 0.83], [z, z], color="#c8ccd2", lw=1)
    ax.add_patch(plt.Rectangle((-0.83, 0), 1.66, 1, fill=False, lw=2.5, color="#3c4043"))
    ax.text(1.25 * (-1 if stand == "L" else 1), 0.5, "우\n타\n자" if stand == "R" else "좌\n타\n자",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#5f6368", linespacing=1.1)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.62, 1.62)
    ax.set_aspect(zh)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color="#5f6368")


def markers(ax, rows, flip, colors=None):
    """번호 달린 큰 점 (같은 자리면 옆으로 나란히)"""
    rows = rows.reset_index(drop=True)
    # 가까운 점끼리 한 묶음으로 → 묶음 안에서 가로로 나란히 (겹치지 않게)
    group = list(range(len(rows)))
    for i in range(len(rows)):
        for j in range(i):
            if abs(rows.at[i, "tx"] - rows.at[j, "tx"]) < 0.35 and abs(rows.at[i, "tz"] - rows.at[j, "tz"]) < 0.2:
                group[i] = group[j]
                break
    for i, r in rows.iterrows():
        same = [j for j in range(len(rows)) if group[j] == group[i]]
        cx = rows.loc[same, "tx"].mean() * flip
        x = cx + (same.index(i) - (len(same) - 1) / 2) * 0.42
        r = r.copy()
        r["tz"] = rows.loc[same, "tz"].mean()
        c = (colors or {}).get(i, COLOR.get(r["pitch_type"], "#888"))
        ax.scatter(x, r["tz"], s=1300 if i == 0 else 900, color=c, edgecolor="white",
                   linewidth=2.5, zorder=5, alpha=0.95 if i == 0 else 0.8)
        ax.text(x, r["tz"], str(i + 1), ha="center", va="center", fontsize=15 if i == 0 else 12,
                fontweight="bold", color="white", zorder=6)


def x_mark(ax, x, z, label, color="#ea4335"):
    ax.scatter(x, z, marker="X", s=380, color=color, zorder=4)
    up = 0.05 <= z <= 0.3                    # 존 아래 테두리 근처면 글자를 위에
    ax.text(x, z + 0.12 if up else z - 0.12, label, ha="center", va="bottom" if up else "top",
            fontsize=9, color=color, fontweight="bold")


def card(color, title, sub, right, line2, line3=""):
    return f"""
<div style="border-left:7px solid {color};background:rgba(127,127,127,.08);border-radius:8px;
            padding:9px 13px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <b>{title} <span style="opacity:.6;font-weight:400">{sub}</span></b><span>{right}</span>
  </div>
  <div style="margin-top:3px">{line2}</div>
  <div style="font-size:.85rem;margin-top:2px;opacity:.75">{line3}</div>
</div>"""


def zone_h(bid):
    z = M["batter_zone"].loc[bid]
    return float(z["sz_top"]), float(z["sz_bot"])


def header(title, pname, bname, stand, balls, strikes, outs, sit, extra_txt=""):
    on = [f"{b}루" for b, v in zip("123", sit[1:]) if v]
    sit_txt = f"{outs}아웃 · " + ("만루" if len(on) == 3 else ", ".join(on) + " 주자" if on else "주자 없음")
    h1, h2 = st.columns([4, 1])
    with h1:
        st.title(title)
        st.markdown(f"#### {pname} &nbsp;vs&nbsp; {bname} ({'우타' if stand == 'R' else '좌타'})")
        st.markdown(f"카운트 **{balls}-{strikes}** · {sit_txt}{extra_txt}")
    with h2:
        st.markdown(scoreboard(balls, strikes, outs, *sit[1:]), unsafe_allow_html=True)


# ---------------------------------------------------------------- 타자 / 투수 공통 탭
@st.cache_resource
def load_hier():
    import scouting as SC
    return SC.load_hier("hier_model.pkl")


def batter_tab(bid, stand):
    prof = M["profile"]
    cell = prof["cell"][prof["cell"]["batter"] == bid].set_index("cell")["b_cell_rv"]
    grid = np.full((5, 5), np.nan)
    for c in range(25):
        grid[c % 5, c // 5] = cell.get(c, prof["cell_league"].get(c, np.nan)) * 100
    if stand == "L":
        grid = grid[:, ::-1]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("핫존 / 콜드존")
        fig, ax = plt.subplots(figsize=(4, 4.4))
        vmax = np.nanmax(np.abs(grid))
        ax.imshow(grid, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((0.5, 0.5), 3, 3, fill=False, lw=2.5, color="black"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("몸쪽 ←   → 바깥쪽" if stand == "L" else "바깥쪽 ←   → 몸쪽")
        ax.set_ylabel("낮게 ←   → 높게")
        ax.set_title("투수 시점 · 빨강 = 타자가 강한 코스", fontsize=10)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("칸 숫자 = 그 칸으로 온 공의 100구당 실점 변화 (+면 타자가 강한 코스). 굵은 선 = 스트라이크존")
    with right:
        st.subheader("타자 성향")
        ov = prof["overall"].set_index("batter")
        d = prof["defaults"]

        def val(c):
            v = ov[c].get(bid, np.nan) if c in ov.columns else np.nan
            return d[c] if pd.isna(v) else v
        st.dataframe(pd.DataFrame([
            ("평균 타구속도 (mph)", f"{val('b_ev'):.1f}", f"{d['b_ev']:.1f}"),
            ("평균 발사각 (°)", f"{val('b_la'):.1f}", f"{d['b_la']:.1f}"),
            ("당겨친 타구", f"{val('b_pull'):.0%}", f"{d['b_pull']:.0%}"),
            ("당겨친 뜬공", f"{val('b_pull_air'):.0%}", f"{d['b_pull_air']:.0%}"),
            ("밀어친 타구", f"{val('b_oppo'):.0%}", f"{d['b_oppo']:.0%}"),
        ], columns=["항목", "이 타자", "리그 평균"]), hide_index=True, width="stretch")
        gp = prof["group"][prof["group"]["batter"] == bid].set_index("group")
        import scouting as SC
        HBW = SC.hier_whiff(load_hier(), bid)
        rows = []
        for g in ["fastball", "breaking", "offspeed"]:
            def gv(c):
                v = gp[c].get(g, np.nan) if c in gp.columns else np.nan
                return d[c][g] if pd.isna(v) else v
            hb = HBW.get(g)
            rows.append({"구종 계열": GROUP_KO[g], "헛스윙률": f"{gv('b_whiff'):.0%}",
                         "헛스윙 [90% 구간] (계층 베이즈)": f"{hb[0]:.0%} [{hb[1]:.0%}~{hb[2]:.0%}] (리그 {hb[3]:.0%})" if hb else "—",
                         "추격률": f"{gv('b_chase'):.0%}", "맞았을 때 타구 품질(xwOBA)": f"{gv('b_xwoba'):.3f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("추격률 = 존 밖 공에 배트를 낸 비율. 표본이 적은 타자는 리그 평균 쪽으로 보정된 값이에요. "
                   "[90% 구간]은 계층 베이즈 모델의 불확실성이에요 (2024·2025) — 2025 검증에서 실제 값이 구간에 86% 들어갔어요.")


def pitcher_tab(pid, pname):
    a = ars_all[ars_all["pitcher"] == pid].sort_values("usage", ascending=False)
    st.subheader(f"{pname}의 구종 (2024)")
    st.dataframe(pd.DataFrame({
        "구종": [ko(t) for t in a["pitch_type"]],
        "구사율": [f"{u:.0%}" for u in a["usage"]],
        "구속(mph)": a["release_speed"].round(1).values,
        "회전수(rpm)": a["release_spin_rate"].round(0).astype(int).values,
        "좌우 무브먼트(in)": a["pfx_x"].round(1).values,
        "수직 무브먼트(in)": a["pfx_z"].round(1).values,
    }), hide_index=True, width="stretch")
    st.caption("무브먼트는 투수 팔 방향 기준으로 통일 (+ = 팔 쪽으로 휨 / 덜 떨어짐). 구사율 4% 미만 구종은 제외.")
    import scouting as SC
    CMD = SC.load_command()
    if CMD is not None and pid in CMD["c"]:
        st.metric("제구 오차 (추정)", SC.command_text(CMD, pid),
                  help="3볼·2-0 카운트 패스트볼(의도가 거의 '스트라이크 넣기' 하나)이 흩어진 정도. 해마다 이어지고(r 0.33) 다음 시즌 볼넷률과 "
                       "관련 있음(r 0.37). 추천 계산엔 리그 공통 가정(사이드바 '제구 오차')을 써요 — 투수별 값을 넣어도 백테스트에서 나아지지 않았어요.")


def spray_fan(summary, league, stand):
    """타구 방향 부채꼴: 땅볼 / 공중 타구별 당김·가운데·밀어침 비율 (포수 시점: 왼쪽 = 3루 쪽)"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    # 포수 시점 각도 (0° = 중견수, +는 1루/우익수 쪽)
    pull_side = -1 if stand == "R" else 1
    sectors = {"당김": (15, 45), "가운데": (-15, 15), "밀어침": (-45, -15)}
    for ax, kind in zip(axes, ["땅볼", "공중 타구"]):
        s, lg = summary[kind], league[kind]
        vmax = 0.7
        for label, (a0, a1) in sectors.items():
            lo, hi = sorted((a0 * pull_side, a1 * pull_side))
            share = s[label]
            diff = share - lg[label]
            color = plt.cm.Reds(0.25 + 0.75 * min(share / vmax, 1))
            ax.add_patch(Wedge((0, 0), 1.0, 90 - hi, 90 - lo, facecolor=color, edgecolor="white", lw=2))
            mid = np.radians(90 - (lo + hi) / 2)
            ax.text(0.62 * np.cos(mid), 0.62 * np.sin(mid), f"{label}\n{share:.0%}",
                    ha="center", va="center", fontsize=11, fontweight="bold")
            ax.text(0.98 * np.cos(mid), 0.98 * np.sin(mid) + 0.08, f"리그 대비 {diff:+.0%}",
                    ha="center", va="bottom", fontsize=8, color="#5f6368")
        ax.add_patch(plt.Polygon([(0, 0), (0.16, 0.16), (0, 0.32), (-0.16, 0.16)], fill=False,
                                 color="#9aa0a6", lw=1.5))
        ax.text(-0.95, 0.02, "3루 쪽", fontsize=9, color="#5f6368")
        ax.text(0.75, 0.02, "1루 쪽", fontsize=9, color="#5f6368")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{kind} ({s['n']}개)", fontsize=12)
    fig.suptitle("포수 시점 타구 방향", fontsize=10, color="#5f6368")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 🛠️ 투수 개선
MECH = "🛠️ 투수 개선"
FT2CM = 30.48


@st.cache_resource
def load_mech():
    """투수 리포트, 투구 샘플, 스윙 모델 — 패스트볼 대비 피처 모델(v2)이 있으면 그걸 씀 (검증에서 더 정확)"""
    if os.path.exists("swing_model_v2.pkl") and os.path.exists("pitch_samples_v2.pkl"):
        return _load("pitcher_report.pkl"), _load("pitch_samples_v2.pkl"), _load("swing_model_v2.pkl")
    return _load("pitcher_report.pkl"), _load("pitch_samples.pkl"), S


@st.cache_resource
def load_move():
    return MC.load_move("release_move.pkl")


@st.cache_data(show_spinner=False)
def run_mech(pid, hand, _t):
    rep, samp, sm = load_mech()
    ph = D["P"][D["P"]["pitcher"] == pid].set_index("pitch_type")
    return MC.simulate(sm, samp, ph, rep, pid, hand, move=load_move())


@st.cache_data(show_spinner=False)
def run_shape(pid, hand, _t):
    rep, samp, sm = load_mech()
    ph = D["P"][D["P"]["pitcher"] == pid].set_index("pitch_type")
    return MC.shape_simulate(sm, samp, ph, rep, pid, hand)


@st.cache_data(show_spinner=False)
def load_val():
    rd = lambda p: pd.read_csv(p) if os.path.exists(p) else None
    return rd("val_release_summary.csv"), rd("val_personal_summary.csv"), rd("val_personal_diff.csv")


def shape_words(t, iv, arm, v):
    w = []
    if iv:
        if t in ("FF", "SI", "FC"):
            w.append(f"수직 {iv * 2.54:+.0f}cm ({'더 떠오르게' if iv > 0 else '덜 떠오르게'})")
        else:
            w.append(f"수직 {iv * 2.54:+.0f}cm ({'덜 떨어지게' if iv > 0 else '더 떨어지게'})")
    if arm:
        w.append(f"가로 {abs(arm) * 2.54:.0f}cm {'팔 쪽으로 더' if arm > 0 else '글러브 쪽으로 더'}")
    if v:
        w.append(f"구속 {v:+.0f}mph")
    return " · ".join(w) or "지금 그대로"


def shape_heatmap(res, t, velo, rec, title):
    g = res[(res["pitch_type"] == t) & (res["d_velo"] == velo)]
    iv_v, arm_v = sorted(g["d_ivb"].unique(), reverse=True), sorted(g["d_arm"].unique())
    val = g.pivot(index="d_ivb", columns="d_arm", values="d").loc[iv_v, arm_v]
    ok = g.pivot(index="d_ivb", columns="d_arm", values="feasible").loc[iv_v, arm_v]
    lim = max(0.5, np.nanmax(np.abs(val.values)))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(val.where(ok.astype(bool)).values, cmap="PRGn", vmin=-lim, vmax=lim)
    for i, a in enumerate(iv_v):
        for j, b in enumerate(arm_v):
            if not ok.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e8eaed"))
                ax.text(j, i, "범위\n밖", ha="center", va="center", fontsize=7, color="#9aa0a6")
            else:
                v_ = val.iloc[i, j]
                ax.text(j, i, "0.0" if abs(v_) < 0.05 else f"{v_:+.1f}", ha="center", va="center", fontsize=9)
            if a == 0 and b == 0 and velo == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#202124", lw=2))
            if rec is not None and a == rec["d_ivb"] and b == rec["d_arm"] and velo == rec["d_velo"]:
                ax.add_patch(plt.Rectangle((j - .45, i - .45), .9, .9, fill=False, ec="#ea4335", lw=2.5))
    ax.set_xticks(range(len(arm_v)), [f"{v * 2.54:+.0f}" for v in arm_v])
    ax.set_yticks(range(len(iv_v)), [f"{v * 2.54:+.0f}" for v in iv_v])
    ax.set_xlabel("← 글러브 쪽으로 더     가로 무브 변화 (cm)     팔 쪽으로 더 →", fontsize=8)
    up = "더 떠오름" if t in ("FF", "SI", "FC") else "덜 떨어짐"
    ax.set_ylabel(f"수직 무브 변화 (cm, +면 {up})", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def val_chart(v, vp=None):
    """검증 그래프: 릴리스를 바꾼 투수들에서 예측 방식별 실제와의 상관 (vp: 개인화 검증 결과가 있으면 v2 숫자로)"""
    sub = v[v["subset"].str.contains("이상 바꾼 경우") & ~v["subset"].str.contains("·")]
    lab = {"P0": "처음 방식\n(릴리스 값을 모델에 직접)", "P4": "지금 방식\n(릴리스→무브→헛스윙)",
           "P3": "공 모양 변화\n(실제 바뀐 무브·구속)"}
    sub = sub[sub["method"].isin(lab)].set_index("method").loc[["P0", "P4", "P3"]].copy()
    if vp is not None:
        q = vp[vp["subset"] == "릴리스 바꾼 경우"].set_index(["kind", "ver"])
        for m_, key in (("P4", ("league", "v2")), ("P3", ("actual", "v2"))):
            if key in q.index:
                sub.loc[m_, ["r", "lo", "hi"]] = q.loc[key, ["r", "lo", "hi"]].values
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(sub))[::-1]
    colors = ["#ea4335", "#fbbc04", "#34a853"]
    ax.barh(y, sub["r"], color=colors, height=0.55)
    ax.errorbar(sub["r"], y, xerr=[sub["r"] - sub["lo"], sub["hi"] - sub["r"]], fmt="none", ecolor="#202124", capsize=4, lw=1)
    ceil = sub["ceiling"].iloc[0]
    ax.axvline(ceil, color="#5f6368", ls="--", lw=1)
    ax.text(ceil, y.max() + 0.45, f" 최대 가능 ≈{ceil:.2f}", fontsize=8, color="#5f6368", va="bottom")
    ax.axvline(0, color="#5f6368", lw=1)
    for yy, (_, r_) in zip(y, sub.iterrows()):
        ax.text(max(r_["hi"], 0) + 0.02, yy, f"{r_['r']:+.2f}", va="center", fontsize=9)
    ax.set_yticks(y, [lab[m] for m in sub.index], fontsize=8)
    ax.set_xlim(-0.3, max(0.7, ceil + 0.1))
    ax.set_ylim(-0.6, y.max() + 0.9)
    ax.set_xlabel("예측한 헛스윙 변화 ↔ 2025 실제 변화 상관 (95% 구간)", fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    return fig, int(sub["n"].iloc[0])


def move_words(dh, ds, de):
    w = []
    if dh:
        w.append(f"타점 {abs(dh) * FT2CM:.0f}cm {'높게' if dh > 0 else '낮게'}")
    if ds:
        w.append(f"팔을 {abs(ds) * FT2CM:.0f}cm " + ("몸에서 더 벌려서 (사이드암 쪽)" if ds > 0 else "몸 쪽으로 세워서 (오버핸드 쪽)"))
    if de:
        w.append(f"익스텐션 {abs(de) * FT2CM:.0f}cm " + ("더 앞에서 놓기" if de > 0 else "덜 끌고 나와서 놓기"))
    return " · ".join(w) or "지금 그대로"


def strengths(r):
    """백분위 80 이상인 장점 문장 (상위 몇 %)"""
    items = []
    names = {"velo_pct": "구속", "ext_pct": "익스텐션", "swstr_pct": "헛스윙 유도", "rv100_pct": "실점 억제",
             "spin_pct": "회전수"}
    for _, x in r[r["n"] >= 100].iterrows():
        for c, nm in names.items():
            if pd.notna(x[c]) and x[c] >= 80:
                items.append((x[c], f"{name_ko(x['pitch_type'])} {nm} 상위 {max(1, round(100 - x[c]))}%"))
    return [t for _, t in sorted(items, reverse=True)[:5]]


def plan_card(title, row, focus_col, focus_name, color):
    if row is None:
        st.markdown(f"**{title}** — 조건에 맞는 조합이 없어요")
        return
    base = row["dh"] == 0 and row["ds"] == 0 and row["de"] == 0
    if not base and row["score"] <= 0:
        st.markdown(f"""<div style="border-left:6px solid {color};background:#f8f9fa;padding:10px 14px;border-radius:6px;margin-bottom:8px">
<div style="font-size:13px;color:#5f6368">{title}</div>
<div style="font-size:17px;font-weight:600;margin:2px 0">이득 볼 만한 변화가 없어요</div>
<div style="font-size:14px;color:#202124">지금 릴리스를 유지하는 게 나아요</div></div>""", unsafe_allow_html=True)
        return
    words = "지금 릴리스가 이미 가장 좋아요 (바꿔도 이득이 거의 없음)" if base else move_words(row["dh"], row["ds"], row["de"])
    big = f"{row[focus_col]:+.1f}%p" if not base else "±0"
    sub = f"{focus_name} 헛스윙 {big} · 전체 {row['d_all']:+.1f}%p" if not base else "다른 조합은 모두 손해이거나 범위 밖"
    st.markdown(
        f"""<div style="border-left:6px solid {color};background:#f8f9fa;padding:10px 14px;border-radius:6px;margin-bottom:8px">
<div style="font-size:13px;color:#5f6368">{title}</div>
<div style="font-size:17px;font-weight:600;margin:2px 0">{words}</div>
<div style="font-size:14px;color:#202124">{sub}</div></div>""", unsafe_allow_html=True)


def mech_heatmap(res, col, de, rec, title):
    g = res[res["de"] == de]
    dh_v, ds_v = sorted(g["dh"].unique(), reverse=True), sorted(g["ds"].unique())
    val = g.pivot(index="dh", columns="ds", values=col).loc[dh_v, ds_v]
    ok = g.pivot(index="dh", columns="ds", values="feasible").loc[dh_v, ds_v]
    lim = max(0.5, np.nanmax(np.abs(val.values)))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    shown = val.where(ok.astype(bool))
    ax.imshow(shown.values, cmap="PRGn", vmin=-lim, vmax=lim)
    for i, h in enumerate(dh_v):
        for j, s_ in enumerate(ds_v):
            if not ok.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e8eaed"))
                ax.text(j, i, "범위\n밖", ha="center", va="center", fontsize=7, color="#9aa0a6")
            else:
                v_ = val.iloc[i, j]
                ax.text(j, i, "0.0" if abs(v_) < 0.05 else f"{v_:+.1f}", ha="center", va="center", fontsize=9)
            if h == 0 and s_ == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#202124", lw=2))
            if rec is not None and h == rec["dh"] and s_ == rec["ds"] and de == rec["de"]:
                ax.add_patch(plt.Rectangle((j - .45, i - .45), .9, .9, fill=False, ec="#ea4335", lw=2.5))
    ax.set_xticks(range(len(ds_v)), [f"{v * FT2CM:+.0f}" for v in ds_v])
    ax.set_yticks(range(len(dh_v)), [f"{v * FT2CM:+.0f}" for v in dh_v])
    ax.set_xlabel("← 팔을 몸 쪽으로 (오버핸드)     옆 위치 (cm)     몸에서 벌려서 (사이드암) →", fontsize=8)
    ax.set_ylabel("타점 높이 변화 (cm)", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def mech_page():
    rep_all, samp_all, sm_ = load_mech()
    is_v2 = "rel" in sm_
    if rep_all is None or samp_all is None or D["P"] is None:
        st.error("pitcher_report.pkl / pitch_samples.pkl / physics.pkl 이 필요해요. "
                 "build_mechanics_data.py 와 build_physics.py 를 먼저 실행하세요.")
        st.stop()
    ok_ids = set(samp_all["pitcher"].unique())
    mp = (rep_all[rep_all["pitcher"].isin(ok_ids)].drop_duplicates("pitcher")
          [["pitcher", "player_name", "p_throws"]].sort_values("player_name"))
    names = mp["player_name"].tolist()
    with st.sidebar:
        st.header("🛠️ 투수 개선")
        pname = st.selectbox("투수", names, index=names.index("Skenes, Paul") if "Skenes, Paul" in names else 0,
                             key="mech_p")
        row = mp[mp["player_name"] == pname].iloc[0]
        pid, hand = int(row["pitcher"]), row["p_throws"]
        r = rep_all[rep_all["pitcher"] == pid].sort_values("usage", ascending=False)
        main = r[r["usage"] >= MC.MAIN_USAGE]
        best_t = r.loc[r["is_best"], "pitch_type"].tolist()
        st.subheader("무엇을 기준으로?")
        opts = ([f"⭐ 제일 좋은 공 살리기 ({name_ko(best_t[0])})"] if best_t else []) + \
               ["📊 전체 구종 합산 (구사율 가중)"] + [f"{ko(t)} 하나만" for t in main["pitch_type"] if t not in best_t[:1]]
        pick = st.radio("기준", opts, label_visibility="collapsed")
        if pick.startswith("⭐"):
            focus = best_t[0]
        elif pick.startswith("📊"):
            focus = "all"
        else:
            focus = pick.split(" ")[0]
        protect = st.checkbox("다른 구종이 손해 보는 조합은 빼기", value=True,
                              help="한 구종을 살리려다 전체(구사율 가중) 헛스윙이 줄어드는 조합은 추천하지 않아요")
        use_move = load_move() is not None

    # ---- 투수 소개
    bio = D["bio"]
    ph = D["P"][D["P"]["pitcher"] == pid]
    arm = np.nanmean(ph["arm_angle"]) if "arm_angle" in ph and ph["arm_angle"].notna().any() else np.nan
    info = [("좌완" if hand == "L" else "우완")]
    if bio is not None and pid in bio.index and pd.notna(bio.loc[pid, "height_in"]):
        hi = bio.loc[pid, "height_in"]
        info.append(f"키 {hi * 2.54:.0f}cm")
    if pd.notna(arm):
        info.append(f"팔 각도 {arm:.0f}°")
    fb = main.iloc[0]
    info.append(f"타점 {fb['rel_z'] * FT2CM:.0f}cm · 익스텐션 {fb['ext'] * FT2CM:.0f}cm ({ko(fb['pitch_type'])} 기준)")
    st.markdown(f"## 🛠️ {pname}")
    st.caption(" · ".join(info))

    # ---- 장점 리포트
    st.subheader("1. 이 투수의 무기")
    good = strengths(r)
    if best_t:
        b = r[r["pitch_type"] == best_t[0]].iloc[0]
        st.markdown(f"⭐ **제일 잘 던지는 공: {ko(best_t[0])}** — 100구당 실점 {b['rv100']:+.2f} "
                    f"(같은 구종 MLB 투수 중 상위 {max(1, round(100 - b['rv100_pct']))}%), 구사율 {b['usage']:.0%}")
    if good:
        st.markdown(" ".join(f"`{g}`" for g in good))
    tab = pd.DataFrame({
        "구종": [("⭐ " if t in best_t else "") + ko(t) for t in r["pitch_type"]],
        "구사율": r["usage"].values * 100,
        "구속(mph)": r["velo"].values,
        "수직 무브(cm)": r["ivb"].values * 2.54,          # ivb는 인치
        "헛스윙%(투구당)": r["swstr"].values * 100,
        "진입각(같은 높이 대비)": r["vaa_rel"].values,
        "구속 백분위": r["velo_pct"].values,
        "헛스윙 백분위": r["swstr_pct"].values,
        "실점 억제 백분위": r["rv100_pct"].values,
    })
    tab = tab[r["usage"].values >= 0.02]
    pc = dict(min_value=0, max_value=100, format="%.0f")
    st.dataframe(tab, hide_index=True, width="stretch", column_config={
        "구사율": st.column_config.NumberColumn(format="%.0f%%"),
        "구속(mph)": st.column_config.NumberColumn(format="%.1f"),
        "수직 무브(cm)": st.column_config.NumberColumn(format="%+.0f"),
        "헛스윙%(투구당)": st.column_config.NumberColumn(format="%.1f"),
        "진입각(같은 높이 대비)": st.column_config.NumberColumn(format="%+.1f°",
                                                         help="+면 같은 높이로 들어오는 같은 계열 공보다 평평하게, -면 가파르게"),
        "구속 백분위": st.column_config.ProgressColumn(**pc),
        "헛스윙 백분위": st.column_config.ProgressColumn(**pc),
        "실점 억제 백분위": st.column_config.ProgressColumn(**pc),
    })
    st.caption("백분위 = 같은 구종을 100구 이상 던진 MLB 투수들 중 위치 (100에 가까울수록 좋음) · 2024 시즌")

    # ---- 2. 공 모양 목표 (검증된 부분)
    st.subheader("2. 🎯 공 모양을 이렇게 바꾸면 — 구종별 목표")
    st.caption("수직 무브·가로 무브·구속을 바꿨을 때 그 구종의 헛스윙 변화. 같은 구종 MLB 투수들이 실제로 던지는 범위 안에서만 골랐어요. "
               "이 부분은 2025 실제 변화로 검증됐어요 (아래 4번).")
    with st.spinner("공 모양 조합 계산 중..."):
        sres, stypes = run_shape(pid, hand, D["t"])
    rr = r.set_index("pitch_type")
    srows, srec = [], {}
    for t in [t for t in r["pitch_type"] if t in stypes]:
        b_ = MC.best_shape(sres, t)
        b_ = b_.iloc[0] if len(b_) else None
        srec[t] = b_
        none = b_ is None or b_["score"] <= 0
        srows.append({"구종": ("⭐ " if t in best_t else "") + ko(t), "구사율": rr.loc[t, "usage"] * 100,
                      "지금 수직cm": rr.loc[t, "ivb"] * 2.54, "지금 가로cm(팔+)": -rr.loc[t, "hb"] * 2.54,
                      "지금 구속": rr.loc[t, "velo"],
                      "추천 변화": "지금 그대로가 좋음" if none else shape_words(t, b_["d_ivb"], b_["d_arm"], b_["d_velo"]),
                      "헛스윙%p": 0.0 if none else b_["d"],
                      "다른 구종%p": 0.0 if none else b_.get("d_others", 0.0)})
    st.dataframe(pd.DataFrame(srows), hide_index=True, width="stretch", column_config={
        "구사율": st.column_config.NumberColumn(format="%.0f%%"),
        "지금 수직cm": st.column_config.NumberColumn(format="%+.0f"),
        "지금 가로cm(팔+)": st.column_config.NumberColumn(format="%+.0f"),
        "지금 구속": st.column_config.NumberColumn(format="%.1f"),
        "헛스윙%p": st.column_config.NumberColumn(format="%+.1f"),
        "다른 구종%p": st.column_config.NumberColumn(
            format="%+.1f", help="주력 패스트볼을 바꾸면 다른 공들의 '패스트볼 대비' 차이가 달라져서 같이 좋아지거나 나빠져요 "
                                 "(다른 구종들의 구사율 가중 평균)"),
    })
    if is_v2:
        fbm = samp_all.loc[samp_all["pitcher"] == pid, "main_fb"].iloc[0]
        st.caption(f"주력 패스트볼({name_ko(fbm)})을 바꾸면 다른 구종도 영향을 받아요: 변화구·체인지업은 패스트볼과 "
                   f"구속·무브 차이가 클수록 대체로 헛스윙이 늘어나서요. 그래서 패스트볼만 좋아지고 다른 공이 손해 보는 조합은 뺐어요.")
    s_def = focus if focus in stypes else r["pitch_type"].iloc[0]
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        t_sel = st.selectbox("자세히 볼 구종", [t for t in r["pitch_type"] if t in stypes],
                             index=[t for t in r["pitch_type"] if t in stypes].index(s_def), format_func=ko, key="shape_t")
    rec_s = srec.get(t_sel)
    v_def = int(rec_s["d_velo"]) if rec_s is not None and rec_s["score"] > 0 else 0
    with sc2:
        v_sel = st.select_slider("구속 변화", options=MC.D_VELO, value=v_def, format_func=lambda v: f"{v:+d}mph",
                                 key="shape_v")
    h1, h2 = st.columns([1, 1])
    with h1:
        fig = shape_heatmap(sres, t_sel, v_sel, rec_s if rec_s is not None and rec_s["score"] > 0 else None,
                            f"{name_ko(t_sel)} 헛스윙 변화 (%p)")
        st.pyplot(fig)
        plt.close(fig)
    with h2:
        vv = sres[(sres["pitch_type"] == t_sel) & (sres["d_ivb"] == 0) & (sres["d_arm"] == 0)].set_index("d_velo")["d"]
        st.markdown(f"**{name_ko(t_sel)} 구속만 바꾸면**")
        st.markdown(" · ".join(f"{int(k):+d}mph → {x:+.1f}%p" for k, x in vv.items() if k != 0))
        if is_v2 and rec_s is not None and rec_s["score"] > 0 and any(c.startswith("x_") for c in sres.columns):
            xs = {c[2:]: rec_s[c] for c in sres.columns if c.startswith("x_") and c[2:] != t_sel and pd.notna(rec_s[c])}
            if any(abs(v_) >= 0.05 for v_ in xs.values()):
                st.markdown(f"**추천대로 바꾸면 다른 구종은**")
                st.markdown(" · ".join(f"{name_ko(u)} {v_:+.1f}%p" for u, v_ in xs.items()))
        st.caption("검은 테두리 = 지금 공, 빨간 테두리 = 추천. 회색 = 같은 구종 MLB 투수들이 안 던지는 모양. "
                   "실제로 이런 공 모양 변화는 그립·손목·회전 조정, 릴리스 조정 등으로 만들어요 (릴리스로 가능한 만큼은 아래 3번).")

    # ---- 3. 릴리스 (체인)
    if not use_move:
        st.warning("release_move.pkl이 없어서 릴리스 변화 계산을 건너뛰었어요. build_release_move.py를 먼저 실행하세요.")
        return
    with st.spinner("릴리스 100가지 조합 시뮬레이션 중..."):
        res, types, usage = run_mech(pid, hand, D["t"])
    if focus != "all" and focus not in types:
        st.warning("이 구종은 샘플이 적어서 계산하지 못했어요. 전체 합산으로 볼게요.")
        focus = "all"
    col = "d_all" if focus == "all" else "d_" + focus
    focus_name = "전체" if focus == "all" else name_ko(focus)

    def first(df):
        return None if df.empty else df.iloc[0]
    rec = first(MC.best_plan(res, focus, types, protect=protect))
    small = first(MC.best_plan(res, focus, types, max_step=0.25, protect=protect))
    cand = MC.best_plan(res, focus, types, protect=protect)
    one = first(cand[(cand[["dh", "ds", "de"]] != 0).sum(axis=1) == 1])

    st.subheader(f"3. 🔧 릴리스로 바꿀 수 있는 만큼 — 기준: {focus_name} 헛스윙")
    st.caption("릴리스를 바꾸면 → 공 모양(무브먼트·구속)이 실제 데이터에서 배운 만큼 바뀌고 → 그 공 모양 변화로 헛스윙을 계산해요. "
               "릴리스 값 자체를 모델에 넣는 방식은 검증에서 틀린 것으로 나와 쓰지 않아요 (4번).")
    c1, c2, c3 = st.columns(3)
    with c1:
        plan_card("🥇 추천 (최대 ±15cm)", rec, col, focus_name, "#ea4335")
    with c2:
        plan_card("🔹 작게 바꾼다면 (±8cm 이내)", small, col, focus_name, "#1a73e8")
    with c3:
        plan_card("☝️ 한 가지만 바꾼다면", one, col, focus_name, "#34a853")
    st.caption("헛스윙 = 투구당 헛스윙 비율(SwStr%). %p는 퍼센트포인트 (예: 15% → 17%면 +2%p). "
               "실제 MLB 투수들이 쓰는 릴리스 범위(구종별 2.5~97.5%) 밖으로 나가는 조합은 제외했어요.")

    # ---- 구종별 득실
    rec_base = rec is not None and rec["dh"] == 0 and rec["ds"] == 0 and rec["de"] == 0
    if rec_base:
        st.caption("지금 릴리스를 유지하는 게 가장 좋아서 구종별 변화표는 생략했어요. 아래 지도에서 다른 조합의 득실을 볼 수 있어요.")
    if rec is not None and not rec_base:
        st.markdown("##### 추천대로 바꾸면 구종별로는?")
        base = res[(res["dh"] == 0) & (res["ds"] == 0) & (res["de"] == 0)].iloc[0]
        rr = r.set_index("pitch_type")
        tt = pd.DataFrame({
            "구종": [("⭐ " if t in best_t else "") + ko(t) for t in types],
            "구사율": [usage[t] * 100 for t in types],
            "실제 헛스윙%": [rr.loc[t, "swstr"] * 100 for t in types],
            "모델 현재": [base[t] * 100 for t in types],
            "바꾼 뒤": [rec[t] * 100 for t in types],
            "변화(%p)": [rec["d_" + t] for t in types],
            "진입각 변화": [rec["vaa_" + t] for t in types],
        })
        if use_move:
            tt["수직 무브 변화(cm)"] = [rec["ivb_" + t] * 2.54 for t in types]
            tt["가로 무브 변화(cm)"] = [-rec["hb_" + t] * 2.54 for t in types]     # +면 팔 쪽으로 더 휨
            tt["구속 변화(mph)"] = [rec["velo_" + t] for t in types]
        cc1 = st.container()
        cc2 = st.columns([2, 3])[0]
        with cc1:
            st.dataframe(tt, hide_index=True, width="stretch", column_config={
                "구사율": st.column_config.NumberColumn(format="%.0f%%"),
                "실제 헛스윙%": st.column_config.NumberColumn(format="%.1f"),
                "모델 현재": st.column_config.NumberColumn(format="%.1f", help="모델이 예측한 지금 릴리스의 헛스윙%"),
                "바꾼 뒤": st.column_config.NumberColumn(format="%.1f"),
                "변화(%p)": st.column_config.NumberColumn(format="%+.2f"),
                "진입각 변화": st.column_config.NumberColumn(format="%+.2f°", help="+면 더 평평하게, -면 더 가파르게 들어옴"),
                "수직 무브 변화(cm)": st.column_config.NumberColumn(format="%+.1f", help="+면 덜 떨어짐(포심은 더 떠오름)"),
                "가로 무브 변화(cm)": st.column_config.NumberColumn(format="%+.1f", help="+면 팔 쪽으로 더 휨"),
                "구속 변화(mph)": st.column_config.NumberColumn(format="%+.1f"),
            })
        with cc2:
            fig, ax = plt.subplots(figsize=(4.2, 0.45 * len(types) + 0.8))
            v = [rec["d_" + t] for t in types]
            ax.barh([name_ko(t) for t in types], v, color=[COLOR.get(t, "#888") for t in types])
            ax.axvline(0, color="#5f6368", lw=1)
            for i, x in enumerate(v):
                ax.text(x, i, f" {x:+.1f} ", va="center", ha="left" if x >= 0 else "right", fontsize=9)
            lim = max(0.5, max(abs(x) for x in v)) * 1.4
            ax.set_xlim(-lim, lim)
            ax.set_xlabel("헛스윙 변화 (%p)", fontsize=9)
            ax.invert_yaxis()
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        # 이유 한 줄
        if focus != "all" and not (rec["dh"] == 0 and rec["ds"] == 0 and rec["de"] == 0):
            st.info(f"💡 이 릴리스 변화로 {name_ko(focus)}의 수직 무브 {rec['ivb_' + focus] * 2.54:+.1f}cm, "
                    f"가로 무브 {-rec['hb_' + focus] * 2.54:+.1f}cm(팔 쪽 +), 구속 {rec['velo_' + focus]:+.1f}mph가 "
                    f"바뀌는 효과예요. 위 2번 표의 목표와 비교해 보면 릴리스로 얻을 수 있는 게 얼마나인지 알 수 있어요.")

    # ---- 지도
    st.markdown("##### 타점 × 옆 위치 지도")
    de_opts = sorted(res["de"].unique())
    de_def = rec["de"] if rec is not None else 0.0
    de_sel = st.select_slider("익스텐션 변화", options=de_opts, value=de_def,
                              format_func=lambda v: f"{v * FT2CM:+.0f}cm")
    m1, m2 = st.columns(2)
    with m1:
        fig = mech_heatmap(res, col, de_sel, rec, f"{focus_name} 헛스윙 변화 (%p)")
        st.pyplot(fig)
        plt.close(fig)
    with m2:
        if focus != "all":
            fig = mech_heatmap(res, "d_all", de_sel, rec, "전체(구사율 가중) 헛스윙 변화 (%p)")
            st.pyplot(fig)
            plt.close(fig)
        st.caption("검은 테두리 = 지금 릴리스, 빨간 테두리 = 추천. 초록일수록 헛스윙 증가, 보라일수록 감소.")

    with st.expander("☝️ 하나씩만 바꾸면? (다른 건 그대로)"):
        lab = {"dh": "타점 높이", "ds": "옆 위치 (+면 사이드암 쪽)", "de": "익스텐션"}
        fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
        for ax, a in zip(axs, ("dh", "ds", "de")):
            o = [x for x in ("dh", "ds", "de") if x != a]
            g = res[(res[o[0]] == 0) & (res[o[1]] == 0)].sort_values(a)
            for t in types:
                ax.plot(g[a] * FT2CM, g["d_" + t], marker="o", color=COLOR.get(t, "#888"), label=name_ko(t),
                        lw=2.5 if t == focus else 1.2)
            ax.axhline(0, color="#9aa0a6", lw=1)
            ax.axvline(0, color="#9aa0a6", lw=1, ls=":")
            ax.set_xlabel(lab[a] + " 변화 (cm)", fontsize=9)
        axs[0].set_ylabel("헛스윙 변화 (%p)", fontsize=9)
        axs[-1].legend(fontsize=8, loc="best")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---- 4. 검증
    st.subheader("4. ✅ 검증: 2024→2025 실제로 릴리스를 바꾼 투수들")
    vs, vp, vd = load_val()
    if vs is None:
        st.caption("val_release_summary.csv가 없어요. validate_release.py prep → eval 을 실행하면 여기에 결과가 나와요.")
    else:
        fig, n_ = val_chart(vs, vp if is_v2 else None)
        v1, v2 = st.columns([3, 2])
        with v1:
            st.pyplot(fig)
            plt.close(fig)
        with v2:
            st.markdown(f"""
2024 데이터만 보고 예측한 헛스윙 변화를, 실제로 릴리스를 8cm 이상 바꾼 {n_}개(투수×구종)의 2025 실제 변화와 비교했어요.

- 🔴 **처음 방식**은 실제와 **관계없음** → 모델이 '타점 높은 투수가 원래 잘 던지더라'(투수 간 차이)를 배운 것이었어요. 그래서 버렸어요.
- 🟢 **공 모양 변화**에 대한 반응은 **검증됨** → 2번 표의 근거.
- 🟡 **지금 릴리스 방식**은 방향은 맞지만 약해요. 릴리스로 설명되는 무브 변화가 실제 변화의 일부(약 40%)뿐이라서요.

점선 = 한 시즌 헛스윙%의 표본 오차 때문에 완벽한 예측이어도 넘을 수 없는 상관.
""")
        if vd is not None and vp is not None:
            q = vp[vp["subset"] == "전체"].set_index(["kind", "ver"])
            dd_ = vd.set_index(["subset", "what"])

            def dtxt(sub, what):
                r_ = dd_.loc[(sub, what)]
                return f"{r_['diff']:+.3f} (95% 구간 {r_['lo']:+.3f} ~ {r_['hi']:+.3f})"
            st.markdown("##### 개인화 검증 (같은 2024→2025 자연 실험, 1,143개 투수×구종)")
            st.markdown(f"""
| 시도 | 결과 | 채택 |
|---|---|---|
| ① 투수 **본인의** 릴리스→무브 관계 (경기별 변화에서 학습, 리그 쪽으로 축소) | 리그 평균 대비 상관 {dtxt("전체", "릴리스: 개인 − 리그 (v1)")} — 차이 없음. 한 투수의 경기별 변화는 너무 작고 흔들려서 개인 관계를 배우기 어려웠어요 | ❌ 리그 평균 유지 |
| ② **자기 패스트볼 대비** 피처 (구속 차·수직/가로 무브 차) | 공 모양 예측 상관 {q.loc[("actual", "v1"), "r"]:.3f} → **{q.loc[("actual", "v2"), "r"]:.3f}**, 차이 {dtxt("전체", "공 모양: v2 − v1")}. 헛스윙 AUC도 체인지업·스플리터에서 0.767 → 0.773 | ✅ 채택 (지금 화면) |
""")

    mv = load_move()
    if mv is not None:
        with st.expander("📐 릴리스를 바꾸면 공은 어떻게 변하나 (실제 데이터에서 배운 관계)"):
            rows_ = []
            for t in types:
                d_up = MC.move_delta(mv, t, 0.5, 0, 0)
                d_side = MC.move_delta(mv, t, 0, 0.5, 0)
                d_ext = MC.move_delta(mv, t, 0, 0, 0.5)
                rows_.append({"구종": ko(t),
                              "타점 15cm↑: 수직무브": d_up["ivb"] * 2.54, "타점 15cm↑: 가로무브": -d_up["hb"] * 2.54,
                              "팔 15cm 벌림: 수직무브": d_side["ivb"] * 2.54, "팔 15cm 벌림: 가로무브": -d_side["hb"] * 2.54,
                              "익스텐션 15cm↑: 구속": d_ext["velo"]})
            st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%+.1f") for c in rows_[0] if c != "구종"})
            st.caption("단위: cm (가로무브 +면 팔 쪽으로 더 휨), 구속 mph. 2024→2025 같은 투수·구종의 평균 변화와 "
                       "한 시즌 안 경기별 변화(구장 차이 제거)를 각각 분석해서 두 결과가 같은 방향일 때만 사용, 크기는 작은 쪽. "
                       "0은 두 방법이 엇갈려 반영하지 않은 것.")

    with st.expander("⚠️ 이 결과를 볼 때 주의할 점"):
        st.markdown("""
- 모든 계산은 그 투수가 2024년에 실제로 던진 공들(코스·카운트·상대 타자 그대로)에서 **공 모양만 바꿔** 다시 예측한 거예요.
- **공 모양 → 헛스윙** 반응은 2025 실제 변화로 검증됐고 크기도 과장되지 않았어요 (실제 ≈ 예측의 1.3배).
  다만 검증 상관이 0.28 수준이라 개별 투수 하나에 대해선 틀릴 수 있어요. 여러 구종·여러 투수에서 평균적으로 맞는 정도예요.
- **릴리스 → 공 모양** 관계는 리그 평균이에요 (예: 포심은 타점이 30cm 높아지면 수직 무브 약 +8cm).
  시즌 간·경기 간 두 방법이 같은 방향일 때만 쓰고 크기는 작은 쪽. 회전수는 엇갈려서 빼었어요. 개인마다 몸이 달라서 실제 변화는 다를 수 있어요.
- 공 모양을 실제로 어떻게 만들지(그립·회전축·손목)는 이 모델이 답하지 않아요. 피치 디자인은 코치·트래킹 장비와 함께 하는 영역이에요.
- **헛스윙만** 봐요. 맞았을 때 타구 질(장타 허용)은 반영하지 않았어요.
- 데이터에서 찾은 **상관관계**지 메카닉 코칭이 아니에요. 몸 상태, 부상 위험, 제구 변화는 고려하지 않아요.
""")
    st.caption("같은 값을 🎯 투구 궤적 모드의 '🔧 릴리스 포인트 바꿔 보기'에 넣으면 궤적이 어떻게 달라지는지 3D로 볼 수 있어요.")


# ---------------------------------------------------------------- 🏏 타자 개선
BAT = "🏏 타자 개선"
SW_INFO = {  # 이름, 단위, 설명
    "bat_speed": ("배트 스피드", "mph", "빠를수록 강한 타구"),
    "swing_length": ("스윙 길이", "ft", "짧을수록 콤팩트"),
    "attack_angle": ("어택 앵글", "°", "+면 올려 치기"),
    "attack_direction": ("스윙 방향", "°", "-면 당기는 방향"),
    "swing_path_tilt": ("스윙 경로 기울기", "°", "클수록 가파른 궤적"),
    "contact_y": ("컨택 포인트", "in", "클수록 몸 앞에서 맞힘"),
}


@st.cache_resource
def load_bat():
    return BM.load("batter_swing.pkl")


@st.cache_data(show_spinner=False)
def run_bat(bid, _t):
    return BM.simulate(load_bat(), bid)


def bat_card(title, row, focus, color, k_len):
    if row is None or (row[["d_bs", "d_aa", "d_cy", "d_dir"]] == 0).all() or row["score"] <= 0:
        words, sub = "지금 스윙 유지가 가장 좋아요", "바꿔서 이득 볼 조합이 없어요"
    elif not row["meaningful"]:
        words = "지금 스타일 유지 (바꿔도 차이 미미)"
        sub = (f"가장 나은 조합도 예측 불확실성보다 작아요 — 참고: {BM.words(row, k_len)} "
               f"(헛스윙 {row['d_miss']:+.1f}%p · 타구 질 {row['d_xwcon']:+.0f} · 스윙당 {row['d_xw_sw']:+.1f})")
    else:
        words = BM.words(row, k_len)
        sub = (f"헛스윙 {row['d_miss']:+.1f}%p · 타구 질 {row['d_xwcon']:+.0f} · 스윙당 {row['d_xw_sw']:+.1f}")
    st.markdown(
        f"""<div style="border-left:6px solid {color};background:#f8f9fa;padding:10px 14px;border-radius:6px;margin-bottom:8px">
<div style="font-size:13px;color:#5f6368">{title}</div>
<div style="font-size:16px;font-weight:600;margin:2px 0;color:#202124">{words}</div>
<div style="font-size:13px;color:#202124">{sub}</div></div>""", unsafe_allow_html=True)


def bat_heatmap(res, col, bs, dr, rec, title, sign=1):
    g = res[(res["d_bs"] == bs) & (res["d_dir"] == dr)]
    aa_v, cy_v = sorted(g["d_aa"].unique(), reverse=True), sorted(g["d_cy"].unique())
    val = g.pivot(index="d_aa", columns="d_cy", values=col).loc[aa_v, cy_v]      # 글자는 실제 값
    ok = g.pivot(index="d_aa", columns="d_cy", values="feasible").loc[aa_v, cy_v]
    lim = max(0.5, np.nanmax(np.abs(val.values)))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow((val * sign).where(ok.astype(bool)).values, cmap="PRGn", vmin=-lim, vmax=lim, aspect="auto")  # 색은 좋은 쪽=초록
    for i, a in enumerate(aa_v):
        for j, c in enumerate(cy_v):
            if not ok.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e8eaed"))
                ax.text(j, i, "범위 밖", ha="center", va="center", fontsize=7, color="#9aa0a6")
            else:
                v_ = val.iloc[i, j]
                ax.text(j, i, "0" if abs(v_) < 0.05 else f"{v_:+.1f}", ha="center", va="center", fontsize=9)
            if a == 0 and c == 0 and bs == 0 and dr == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#202124", lw=2))
            if rec is not None and a == rec["d_aa"] and c == rec["d_cy"] and bs == rec["d_bs"] and dr == rec["d_dir"]:
                ax.add_patch(plt.Rectangle((j - .45, i - .45), .9, .9, fill=False, ec="#ea4335", lw=2.5))
    ax.set_xticks(range(len(cy_v)), [f"{v * 2.54:+.0f}" for v in cy_v])
    ax.set_yticks(range(len(aa_v)), [f"{v:+g}°" for v in aa_v])
    ax.set_xlabel("← 뒤에서     컨택 포인트 변화 (cm)     앞에서 →", fontsize=8)
    ax.set_ylabel("어택 앵글 추가 변화 (컨택 이동분 별도)", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def bat_page():
    BS = load_bat()
    if BS is None:
        st.error("batter_swing.pkl 이 필요해요. build_batter_swing.py val → save 를 먼저 실행하세요.")
        st.stop()
    rep = BS["report"]
    names = M["batter_names"]
    opts = sorted([(names.get(b, f"ID {b}"), b) for b in rep.index], key=lambda x: (x[0].startswith("ID"), x[0]))
    labels = [o[0] for o in opts]
    with st.sidebar:
        st.header("🏏 타자 개선")
        nm = st.selectbox("타자", labels, index=labels.index("Soto, Juan") if "Soto, Juan" in labels else 0, key="bat_p")
        bid = dict(opts)[nm]
        st.subheader("무엇을 기준으로?")
        focus = st.radio("기준", list(BM.FOCUS), format_func=lambda k: BM.FOCUS[k], label_visibility="collapsed")
        protect = st.checkbox("종합 가치가 떨어지는 조합은 빼기", value=True,
                              help="타구 질이나 헛스윙만 보다가 스윙당 xwOBA가 1포인트 넘게 떨어지는 조합은 추천하지 않아요")
    r = rep.loc[bid]
    arch = BM.archetype(r)
    st.markdown(f"## 🏏 {nm}")
    st.caption(f"{'좌타' if r['stand'] == 'L' else '우타'} · 2025 시즌 스윙 {int(r['n']):,}개 (배트 트래킹) · "
               f"유형: **{arch}** (배트 스피드 백분위 {r['bat_speed_pct']:.0f})")
    if focus == "style":
        st.info(f"⭐ {arch} 기준: {BM.STYLE_RULE[arch]['desc']}. "
                "실제 데이터에서 어택 앵글을 올렸을 때 컨택형은 남는 게 거의 없고(헛스윙 증가 ≈ 타구 질 증가), "
                "파워형은 타구 질이 두 배 넘게 올라서 이득이 남았어요 (아래 3번).")

    # ---- 1. 스윙 리포트
    st.subheader("1. 이 타자의 스윙")
    rows_ = []
    for c, (lab, unit, desc) in SW_INFO.items():
        v = r[c] * (30.48 if unit == "ft" else 2.54 if unit == "in" else 1)
        u = "cm" if unit in ("ft", "in") else unit
        rows_.append({"항목": lab, "값": f"{v:.1f}{u}", "설명": desc, "MLB 백분위": r[c + "_pct"]})
    cA, cB = st.columns([3, 2])
    with cA:
        st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch", column_config={
            "MLB 백분위": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f",
                                                       help="2025 시즌 200스윙 이상 타자 중 위치 (높을수록 값이 큼)")})
        st.caption("컨택 포인트는 '타자 몸 기준 몇 cm 앞'이라, 체격이 크고 팔이 긴 타자는 같은 타이밍이어도 더 앞으로 찍힐 수 있어요 "
                   "(예: 데라크루즈는 97백분위인데 당긴 타구 비율은 리그 평균 이하). 다른 타자와 비교할 땐 스윙 방향·당긴 타구 비율도 같이 보세요.")
    with cB:
        st.markdown(f"""
| 결과 | 값 | 백분위 |
|---|---|---|
| 헛스윙% (스윙당) | {r['whiff'] * 100:.1f}% | {r['whiff_pct']:.0f} (높을수록 덜 헛스윙) |
| 타구 질 (인플레이 xwOBA) | {r['xwcon']:.3f} | {r['xwcon_pct']:.0f} |
| 스윙당 xwOBA | {r['xw_sw']:.3f} | {r['xw_sw_pct']:.0f} |
""")
        st.caption("xwOBA = 타구 속도·각도로 계산한 기대 가치 (운 빼고 '얼마나 잘 맞혔나'). 스윙당 = 헛스윙·파울은 0으로 계산")

    # ---- 2. 스윙 목표
    with st.spinner("스윙 180가지 조합 시뮬레이션 중..."):
        res = run_bat(bid, D["t"])
    first = lambda df: None if df.empty else df.iloc[0]
    rec = first(BM.best(res, focus, protect=protect, arch=arch))
    nobs = first(BM.best(res, focus, only=["d_aa", "d_cy", "d_dir"], protect=protect, arch=arch))
    one = None
    for ax_ in ("d_aa", "d_cy", "d_dir", "d_bs"):
        c_ = first(BM.best(res, focus, only=[ax_], protect=protect, arch=arch))
        if c_ is not None and (one is None or c_["score"] > one["score"]):
            one = c_
    st.subheader(f"2. 🎯 스윙을 이렇게 바꾸면 — 기준: {BM.FOCUS[focus]}" + (f" → {arch}" if focus == "style" else ""))
    st.caption("그 타자가 2025년에 실제로 상대한 공들(코스·카운트·구종 그대로)에서 스윙 스타일(시즌 평균)만 바꿔 다시 예측. "
               "바꾸는 폭은 실제 타자들이 한 시즌 사이에 바꾼 정도(상위 10%)까지, 바꾼 뒤 값은 MLB 타자 범위(2.5~97.5%) 안에서만 골랐어요.")
    k1, k2, k3 = st.columns(3)
    with k1:
        bat_card("🥇 추천", rec, focus, "#ea4335", BS["k_len"])
    with k2:
        bat_card("🔹 배트 스피드는 그대로 두고", nobs, focus, "#1a73e8", BS["k_len"])
    with k3:
        bat_card("☝️ 한 가지만 바꾼다면", one, focus, "#34a853", BS["k_len"])
    st.caption("타구 질·스윙당 단위 = xwOBA 포인트 (0.001). 예: +10이면 .350 → .360. "
               f"'차이 미미' = 이득이 스윙당 {BM.MEANINGFUL['xw_sw']:g} / 타구 질 {BM.MEANINGFUL['xwcon']:g} 포인트 / "
               f"헛스윙 {BM.MEANINGFUL['miss']}%p 미만 — 타자 성적이 한 시즌 사이 원래 출렁이는 폭의 1/3도 안 돼서 실제로 티가 안 나는 수준")

    if rec is not None and not (rec[["d_bs", "d_aa", "d_cy", "d_dir"]] == 0).all() and rec["meaningful"]:
        st.markdown("##### 추천대로 바꾸면")
        m1, m2 = st.columns([3, 2])
        with m1:
            t = pd.DataFrame({
                "결과": ["헛스윙% (스윙당)", "맞혔을 때 인플레이%", "타구 질 (인플레이 xwOBA)", "스윙당 xwOBA"],
                "지금": [f"{r['whiff'] * 100:.1f}%", f"{r['bip_rate'] * 100:.1f}%", f"{r['xwcon']:.3f}", f"{r['xw_sw']:.3f}"],
                "변화": [f"{rec['d_miss']:+.1f}%p", f"{rec['d_bip']:+.1f}%p", f"{rec['d_xwcon']:+.0f}", f"{rec['d_xw_sw']:+.1f}"],
                "바꾼 뒤(예상)": [f"{r['whiff'] * 100 + rec['d_miss']:.1f}%", f"{r['bip_rate'] * 100 + rec['d_bip']:.1f}%",
                               f"{r['xwcon'] + rec['d_xwcon'] / 1000:.3f}", f"{r['xw_sw'] + rec['d_xw_sw'] / 1000:.3f}"],
            })
            st.dataframe(t, hide_index=True, width="stretch")
            if rec["d_bs"]:
                st.caption(f"배트 스피드 {r['bat_speed']:.1f} → {rec['new_bs']:.1f}mph, 스윙 길이 "
                           f"{r['swing_length'] * 30.48:.0f} → {rec['new_len'] * 30.48:.0f}cm "
                           f"(실제 타자들은 배트 스피드 1mph 올리면 스윙이 평균 {BS['k_len'] * 30.48:.1f}cm 길어졌어요)")
        with m2:
            hz = ["낮게", "가운데", "높게"]
            v = [rec["h_" + z] for z in hz]
            fig, ax = plt.subplots(figsize=(3.8, 2.4))
            ax.bar(hz, v, color=["#1a73e8" if x >= 0 else "#ea4335" for x in v])
            ax.axhline(0, color="#5f6368", lw=1)
            for i, x in enumerate(v):
                ax.text(i, x, f"{x:+.1f}", ha="center", va="bottom" if x >= 0 else "top", fontsize=9)
            ax.set_title("공 높이별 스윙당 xwOBA 변화", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("같은 스윙 변화라도 공 높이에 따라 득실이 달라요 (그 타자가 실제로 상대한 공 기준)")

    st.markdown("##### 어택 앵글 × 컨택 포인트 지도")
    g1, g2 = st.columns(2)
    with g1:
        bs_sel = st.select_slider("배트 스피드 변화", options=BM.D_BS, value=float(rec["d_bs"]) if rec is not None else 0,
                                  format_func=lambda v: f"{v:+g}mph", key="bat_bs")
    with g2:
        dr_sel = st.select_slider("스윙 방향 변화", options=BM.D_DIR, value=int(rec["d_dir"]) if rec is not None else 0,
                                  format_func=lambda v: f"{v:+d}° ({'당기는 쪽' if v < 0 else '미는 쪽' if v > 0 else '그대로'})",
                                  key="bat_dr")
    h1, h2, h3 = st.columns(3)
    for col_, (c, lab, sg) in zip((h1, h2, h3), (("d_miss", "헛스윙 변화 (%p, -면 줄어듦 = 초록)", -1),
                                                 ("d_xwcon", "타구 질 변화 (xwOBA 포인트)", 1),
                                                 ("d_xw_sw", "스윙당 xwOBA 변화", 1))):
        with col_:
            fig = bat_heatmap(res, c, bs_sel, dr_sel, rec, lab, sign=sg)
            st.pyplot(fig)
            plt.close(fig)
    st.caption("검은 테두리 = 지금 스윙, 빨간 테두리 = 추천. 세 지도를 같이 보면 '올려 치면 타구 질은 좋아지지만 헛스윙이 는다' 같은 득실이 보여요. "
               f"컨택 포인트를 옮기면 어택 앵글과 스윙 방향도 자연히 따라와요 (1in 앞으로 → 어택 앵글 {BS.get('k_aa_cy', 0):+.2f}°, "
               f"방향 {BS.get('k_dir_cy', 0):+.2f}° — 실제 타자 286명의 시즌 간 변화). 세로축은 그것에 더해 '추가로' 바꾸는 어택 앵글이에요.")

    # ---- 3. 검증
    st.subheader("3. ✅ 검증: 2024→2025 실제로 스윙을 바꾼 타자들")
    vb = pd.read_csv("val_batter_summary.csv") if os.path.exists("val_batter_summary.csv") else None
    if vb is None:
        st.caption("val_batter_summary.csv 가 없어요 (build_batter_swing.py val).")
    else:
        q = vb[vb["method"] == "F 스타일+조합"].set_index("target").loc[["whiff", "xwcon", "xw_sw"]]
        fig, ax = plt.subplots(figsize=(6, 2.4))
        y = np.arange(3)[::-1]
        ax.barh(y, q["r"], color="#34a853", height=0.55)
        ax.errorbar(q["r"], y, xerr=[q["r"] - q["lo"], q["hi"] - q["r"]], fmt="none", ecolor="#202124", capsize=4, lw=1)
        for yy, (_, rr_) in zip(y, q.iterrows()):
            ax.text(rr_["hi"] + 0.02, yy, f"{rr_['r']:+.2f}", va="center", fontsize=9)
        ax.set_yticks(y, ["헛스윙%", "타구 질", "스윙당 xwOBA"], fontsize=9)
        ax.axvline(0, color="#5f6368", lw=1)
        ax.set_xlim(-0.1, 0.8)
        ax.set_xlabel("스윙 변화로 예측한 결과 변화 ↔ 실제 변화 상관 (95% 구간)", fontsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.tight_layout()
        v1_, v2_ = st.columns([3, 2])
        with v1_:
            st.pyplot(fig)
            plt.close(fig)
        with v2_:
            st.markdown(f"""
두 시즌 모두 300스윙 이상 한 타자 **{int(q['n'].iloc[0])}명**의 2024→2025 스윙 변화를 모델에 넣어 예측한 결과 변화와, 실제 결과 변화를 비교했어요.

- 투수 릴리스(상관 −0.06)와 달리 **스윙 변화는 결과 변화를 꽤 잘 맞혀요.**
- 헛스윙 {q.loc['whiff', 'r']:.2f}, 타구 질 {q.loc['xwcon', 'r']:.2f}, 스윙당 {q.loc['xw_sw', 'r']:.2f} (투수 공 모양 검증은 0.29).
- 처음 만든 GBM 모델은 상관은 비슷했지만 '이 타자 = 이 점'을 외워서 지금 스윙이 항상 최적처럼 나왔어요.
  그래서 공 상황은 GBM, 스윙 스타일은 매끄러운 식(+ 배트 스피드×어택 앵글 같은 조합)으로 나눈 모델로 바꿨어요.
""")
    vg = pd.read_csv("val_batter_groups.csv") if os.path.exists("val_batter_groups.csv") else None
    if vg is not None:
        st.markdown("##### 타자 유형별: 어택 앵글을 1° 올리면 (실제 2024→2025 변화 vs 모델)")
        tb = vg.pivot(index="group", columns="target", values="actual")
        tf = vg.pivot(index="group", columns="target", values="F")
        order = ["컨택형(느린 배트)", "중간", "파워형(빠른 배트)"]
        show = pd.DataFrame({
            "유형": order,
            "헛스윙 실제": [f"{tb.loc[g, 'miss']:+.1f}%p" for g in order],
            "헛스윙 모델": [f"{tf.loc[g, 'miss']:+.1f}%p" for g in order],
            "타구 질 실제": [f"{tb.loc[g, 'xwcon']:+.1f}" for g in order],
            "타구 질 모델": [f"{tf.loc[g, 'xwcon']:+.1f}" for g in order],
            "스윙당 실제": [f"{tb.loc[g, 'xw_sw']:+.1f}" for g in order],
            "스윙당 모델": [f"{tf.loc[g, 'xw_sw']:+.1f}" for g in order],
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("배트 스피드 기준 3등분, 각 약 95명. 모델은 헛스윙 쪽은 잘 맞히고, 유형별 차이 방향(파워형일수록 타구 질 이득이 큼)도 "
                   "잡지만 파워형의 타구 질 이득은 실제(+12)보다 작게(+5) 봐요. 그래서 유형별 기준(⭐)을 따로 두었어요.")

    with st.expander("⚠️ 이 결과를 볼 때 주의할 점"):
        st.markdown("""
- 검증은 '스윙이 바뀐 시즌에 결과도 바뀌었나'예요. 스윙 변화와 결과 변화가 **같은 시즌에 함께 측정**돼서, 예를 들어 몸 상태가 좋아져 둘 다 좋아진 경우도 섞여 있어요.
- **스윙당 xwOBA는 헛스윙과 파울을 0으로 계산**해서, 헛스윙을 줄이는 쪽(평평한 스윙)을 조금 더 좋게 봐요. 장타력이 무기인 타자는 '타구 질' 기준도 같이 보세요.
- 배트 스피드는 힘·근력과 관련이 있어서 원한다고 바로 올릴 수 있는 값이 아니에요.
- 레그킥, 그립, 손 위치 같은 **타격폼 자체는 공개 데이터에 없어서** 다루지 않아요. 여기 나온 건 '어떤 스윙 결과값을 목표로 할지'예요.
- 모델은 리그 전체 타자들에게서 배운 관계예요. 개인마다 다른 부분은 지금 스윙 값(2차식)과 그 타자가 상대한 공의 코스로만 반영돼요.
""")


# ---------------------------------------------------------------- 🧬 선수 지도 (임베딩)
EMBM = "🧬 선수 지도"


@st.cache_resource
def load_emb():
    return EM.load("embed_model.npz")


def emb_name(kind, pid, st_row=None):
    if kind == "batter":
        return M["batter_names"].get(int(pid), f"ID {int(pid)}")
    if st_row is not None and isinstance(st_row.get("name"), str) and st_row.get("name") not in ("", "nan"):
        return st_row["name"]
    return f"ID {int(pid)}"


def emb_page():
    T = load_emb()
    if T is None:
        st.error("embed_model.npz 가 필요해요 (build_embed.py save).")
        st.stop()
    with st.sidebar:
        st.header("🧬 선수 지도")
        kind = st.radio("누구를 볼까", ["batter", "pitcher"], format_func=lambda k: "타자" if k == "batter" else "투수",
                        horizontal=True)
        min_n = st.slider("최소 투구 수 (2024+2025)", 300, 3000, 1500 if kind == "batter" else 1000, 100,
                          help="표본이 적은 선수는 임베딩이 리그 평균 쪽으로 당겨져 있어서 지도에서 빼는 게 보기 좋아요")
        df, var, axes = EM.pca_map(T, kind, min_n)
        df["name"] = [emb_name(kind, i, r) for i, r in zip(df["id"], df.to_dict("records"))]
        names = sorted(df["name"])
        default = "Judge, Aaron" if kind == "batter" else "Skenes, Paul"
        pick = st.selectbox("선수", names, index=names.index(default) if default in names else 0, key=f"emb_{kind}")
        color = st.selectbox("색으로 보기", [c for c in ["whiff", "chase", "swing", "ev", "rv100", "velo", "fb_share"]
                                         if c in df.columns], format_func=lambda c: EM.STAT_KO.get(c, c))
    pid = int(df.loc[df["name"] == pick, "id"].iloc[0])
    sim = EM.similar(T, kind, pid, k=10, min_n=min_n)
    st.markdown(f"## 🧬 {'타자' if kind == 'batter' else '투수'} 지도 — {pick}")
    st.caption("신경망이 2024~2025 투구 140만 개를 보며 선수마다 16개 숫자(임베딩)를 스스로 배웠어요. "
               "숫자의 뜻을 사람이 정해 준 게 아니라, '이 선수가 이런 공에 어떻게 반응하나'를 맞히려고 학습하다 생긴 성향 벡터예요. "
               "가까이 있는 선수일수록 공에 반응하는 방식이 비슷해요.")
    if kind == "pitcher":
        st.info("투수 임베딩은 구속·무브먼트·코스 같은 공 정보를 모델에 이미 따로 넣었기 때문에, **그걸로 설명 안 되는 나머지**"
                "(디셉션, 공이 실제보다 좋아/나빠 보이는 정도, 피치 터널 등)를 담아요. 그래서 '구종 구성이 비슷한 투수'와는 달라요. "
                "구종 구성 비교는 🛠️ 투수 개선의 리포트를 보세요.")
    near = set(sim["id"]) if sim is not None else set()
    df["그룹"] = np.where(df["id"] == pid, "선택", np.where(df["id"].isin(near), "비슷한 10명", "기타"))
    try:
        import plotly.express as px
        fig = px.scatter(df, x="pc1", y="pc2", color=color, hover_name="name",
                         hover_data={c: ":.3f" for c in ["whiff", "chase", "ev", "rv100", "velo"] if c in df.columns},
                         color_continuous_scale="RdBu_r", opacity=0.55, height=560)
        hi = df[df["그룹"] != "기타"]
        fig.add_scatter(x=hi["pc1"], y=hi["pc2"], mode="markers+text", text=hi["name"].str.split(",").str[0],
                        textposition="top center", marker=dict(size=[16 if g == "선택" else 10 for g in hi["그룹"]],
                                                               color=["#ea4335" if g == "선택" else "#202124" for g in hi["그룹"]],
                                                               symbol=["star" if g == "선택" else "circle" for g in hi["그룹"]]),
                        showlegend=False, hoverinfo="skip")
        fig.update_layout(xaxis_title=f"축 1 ({var[0]:.0%}) — {EM.axis_label(axes[0])}",
                          yaxis_title=f"축 2 ({var[1]:.0%}) — {EM.axis_label(axes[1])}",
                          coloraxis_colorbar_title=EM.STAT_KO.get(color, color), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        st.scatter_chart(df, x="pc1", y="pc2", color=color)
    st.caption("16차원을 2차원으로 눌러 그린 그림이라 실제 거리와 조금 달라요. 축 이름은 그 축과 가장 상관이 큰 기본 지표로 자동으로 붙였어요. "
               "임베딩을 학습할 때 이 지표들을 직접 넣지 않았는데도 축이 이렇게 정리됐다는 게 포인트예요.")
    if sim is not None:
        st.markdown(f"##### {pick}와 가장 비슷한 {'타자' if kind == 'batter' else '투수'} 10명")
        s = sim.copy()
        s.insert(0, "선수", [emb_name(kind, i, r) for i, r in zip(s["id"], s.to_dict("records"))])
        cols = {"선수": "선수", "sim": "유사도"}
        for c in (["whiff", "chase", "ev", "rv100"] if kind == "batter" else ["velo", "whiff", "fb_share", "rv100"]):
            if c in s.columns:
                cols[c] = EM.STAT_KO[c]
        s = s[list(cols)].rename(columns=cols)
        me = df[df["id"] == pid].iloc[0]
        st.dataframe(s, hide_index=True, width="stretch", column_config={
            "유사도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            **{EM.STAT_KO[c]: st.column_config.NumberColumn(format="%.3f" if c != "velo" else "%.1f")
               for c in ["whiff", "chase", "ev", "rv100", "velo", "fb_share"] if c in cols}})
        st.caption(f"참고 — {pick}: " + " · ".join(f"{EM.STAT_KO[c]} {me[c]:.3f}" for c in ["whiff", "chase", "ev", "velo"]
                                                   if c in me and pd.notna(me[c])))
    st.subheader("✅ 검증: 기존 모델보다 잘 맞히나? (2024 학습 → 2025 예측)")
    vs = pd.read_csv("val_embed_summary.csv") if os.path.exists("val_embed_summary.csv") else None
    if vs is not None:
        q = vs.pivot(index="bucket", columns="model", values="whiff_auc")
        order = [b for b in ["전체", "2024 표본 0 (신인)", "2024 1~500구", "2024 500~1500구", "2024 1500구+"] if b in q.index]
        cols_ = [c for c in ["LightGBM+프로필", "신경망+임베딩", "신경망(임베딩 끔)", "둘 평균"] if c in q.columns]
        t = q.loc[order, cols_].reset_index().rename(columns={"bucket": "타자 (2024 표본)"})
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in cols_})
        st.markdown("""
헛스윙 예측 AUC (높을수록 정확, 2025 시즌 70만 구로 평가).
- **신경망+임베딩이 기존 LightGBM+타자 프로필보다 전체적으로 더 정확**해요. 특히 표본이 충분한 타자(500구 이상)에서 차이가 커요.
- **임베딩을 끄면(모두 리그 평균) 크게 떨어져요** → 임베딩이 선수 성향을 실제로 담고 있다는 뜻.
- 표본이 거의 없는 타자(신인, 500구 미만)는 기존 방식과 비슷하거나 약간 못해요. 그래서 **두 모델 평균**이 모든 구간에서 가장 좋아요.
- 하이퍼파라미터(학습 횟수·축소 강도)는 2024 안에서만(4~7월 학습 → 8~9월) 골라서, 2025 결과는 처음 보는 데이터로 평가한 거예요.
""")


# ---------------------------------------------------------------- ♟️ 게임이론 볼배합
GAMEM = "♟️ 게임이론 볼배합"


@st.cache_resource
def load_game():
    return GM.load("game_mix.pkl", "game_exploit.pkl")


@st.cache_resource
def load_minimax():
    return GM.load_minimax("minimax.pkl")


def minimax_section(pid, pname):
    mm = load_minimax()
    if mm is None:
        return
    st.subheader("⚖️ MLB 투수들은 미니맥스대로 던지나?")
    st.caption("미니맥스(제로섬 게임의 균형)라면 투수가 실제로 쓰는 구종들은 같은 상황에서 결과가 같아야 해요 (무차별 원리). "
               "같은 투수·같은 카운트·같은 좌우 대결 안에서만 비교해서, 변화구·오프스피드가 패스트볼보다 100구당 몇 점 더/덜 내줬는지 봤어요.")
    L = mm["league"]
    t = pd.DataFrame([{"시즌": yr, "투구": f"{L[yr]['n']:,}", "투수": L[yr]["pitchers"],
                       "변화구 − 패스트볼": f"{L[yr]['g1'][0]:+.2f} [{L[yr]['g1'][1][0]:+.2f}, {L[yr]['g1'][1][1]:+.2f}]",
                       "오프스피드 − 패스트볼": f"{L[yr]['g2'][0]:+.2f} [{L[yr]['g2'][1][0]:+.2f}, {L[yr]['g2'][1][1]:+.2f}]"}
                      for yr in (2024, 2025)])
    st.dataframe(t, hide_index=True, width="stretch")
    st.markdown("""
- 전체로 보면 차이가 **100구당 0.1~0.3점**으로 아주 작아요 → MLB 투수들은 전체적으로 **미니맥스에 꽤 가깝게** 던져요.
  (2000년대 데이터로는 '패스트볼을 너무 많이 던진다'는 연구(Kovash & Levitt, 2009)가 있었는데, 지금은 오히려 변화구 쪽이 아주 약간 더 내줘요.)
""")
    rows = []
    for yr in (2024, 2025):
        bc = mm["by_count"][yr]
        for g, lab in (("g1", "변화구"), ("g2", "오프스피드")):
            for _, r_ in bc.iterrows():
                rows.append({"yr": yr, "count": r_["count"], "g": lab, "v": r_[g], "lo": r_[g + "_lo"], "hi": r_[g + "_hi"]})
    df = pd.DataFrame(rows)
    order = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-1", "3-2"]
    df = df[df["count"].isin(order)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, lab in zip(axs, ("변화구", "오프스피드")):
        for k, (yr, colr) in enumerate(((2024, "#9aa0a6"), (2025, "#1a73e8"))):
            s = df[(df["g"] == lab) & (df["yr"] == yr)].set_index("count").reindex(order)
            x = np.arange(len(order)) + (k - 0.5) * 0.3
            ax.errorbar(x, s["v"], yerr=[s["v"] - s["lo"], s["hi"] - s["v"]], fmt="o", color=colr, ms=4, capsize=2,
                        lw=1, label=str(yr))
        ax.axhline(0, color="#5f6368", lw=1)
        ax.set_xticks(range(len(order)), order, fontsize=8)
        ax.set_title(f"{lab} − 패스트볼 (100구당 실점, 음수 = 패스트볼보다 좋음)", fontsize=9)
        ax.set_ylim(-3, 3.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    rp = GM.replicated(mm)
    if len(rp):
        lines = " / ".join(f"{r_['count']} {r_['group']} ({r_['v24']:+.2f} → {r_['v25']:+.2f})" for _, r_ in rp.iterrows())
        st.success(f"두 시즌 모두 확실하게(95% 구간이 0을 안 걸침) 같은 방향으로 나온 건: **{lines}** — "
                   "예: 0-2에서 체인지업·스플리터가 패스트볼보다 100구당 약 1점 덜 내줘요. "
                   f"미니맥스대로라면 그 카운트에서 해당 계열을 더 던질 여지가 있다는 뜻이에요 (그 카운트 패스트볼 비율 약 {rp['fb_share'].iloc[0]:.0%}).")
    st.caption("3-0은 거의 패스트볼만 던져서 뺐고, 3-1 오프스피드는 표본이 적어 구간이 매우 넓어요. 같은 카운트 안의 다른 상황"
               "(주자·타자 등) 차이까지 완전히 통제하진 못해요.")
    pg = GM.pitcher_gap(mm, pid)
    per = mm["persist"]
    if pg is not None:
        side = "패스트볼이 다른 공보다 더 내줌 → 패스트볼을 조금 줄일 여지" if pg["gap"] > 0.1 else \
               "패스트볼이 다른 공보다 덜 내줌 → 패스트볼을 조금 늘릴 여지" if pg["gap"] < -0.1 else "거의 균형 (미니맥스에 가까움)"
        st.markdown(f"**{pname}**: 같은 상황에서 '패스트볼 − 나머지' = **{pg['gap']:+.2f}** 점/100구 (축소 보정, 원래 {pg['raw']:+.2f} ± {pg['se']:.2f}) — {side}. "
                    f"패스트볼 비율 {pg['fb_share']:.0%}.")
    st.caption(f"⚠️ 투수 한 명 단위 차이는 거의 운이에요: 2024 차이와 2025 차이의 상관이 {per['r_raw']:+.2f} ({per['n']}명). "
               f"다만 2024에 패스트볼이 상대적으로 나빴던 투수들이 2025에 패스트볼을 약간 줄이는 경향은 있었어요 (상관 {per['fb_change']:+.2f}) "
               "— 투수들이 스스로 균형 쪽으로 조정하는 모습이에요.")


def game_page():
    gmix, gexp = load_game()
    beta0 = float(gmix["beta"]) if gmix else GM.DEFAULT_BETA
    pnames, bnames = sorted(pitcher_names), sorted(batters)
    with st.sidebar:
        st.header("♟️ 게임이론 볼배합")
        pname = st.selectbox("투수", pnames, index=pnames.index("Skenes, Paul") if "Skenes, Paul" in pnames else 0, key="g_p")
        bname = st.selectbox("타자", bnames, index=bnames.index("Judge, Aaron") if "Judge, Aaron" in bnames else 0, key="g_b")
        c1, c2 = st.columns(2)
        balls = c1.selectbox("볼", [0, 1, 2, 3], index=1, key="g_ball")
        strikes = c2.selectbox("스트라이크", [0, 1, 2], index=1, key="g_str")
        outs = st.selectbox("아웃", [0, 1, 2], key="g_out")
        runners = st.multiselect("주자", [1, 2, 3], format_func=lambda b: f"{b}루", key="g_run")
        beta = st.slider("타자가 얼마나 노리고 들어오나 (β)", 0.0, 60.0, beta0, 1.0,
                         help=f"0 = 타자가 투수 성향을 전혀 안 읽음 → 제일 좋은 공만 던지는 게 최선. "
                              f"기본값 {beta0:g} = MLB 투수들의 실제 카운트별 선택에 가장 잘 맞는 값")
    pid, bid = pitcher_names[pname], batters[bname]
    s = M["batter_stand"].get(bid, "R")
    p_hand = ars_all.loc[ars_all["pitcher"] == pid, "p_throws"].iloc[0]
    stand = s if s in ("R", "L") else ("L" if p_hand == "R" else "R")
    sit = (outs, int(1 in runners), int(2 in runners), int(3 in runners))
    with st.spinner("구종·코스별 기대 실점 계산 중..."):
        res = run_defense(pid, bid, balls, strikes, stand, 0.5, None, sit, D["t"])
    df, ev = GM.plan(res, beta)
    st.markdown(f"## ♟️ {pname} → {bname} · {balls}-{strikes}")
    st.caption("'제일 좋은 공'만 계속 던지면 타자가 그 공을 노리고 들어와요. 그래서 실제 투수들은 섞어 던져요. "
               "여기서는 구종별 기대 실점(매치업 모델)과 '많이 던질수록 노림당하는 정도(β)'로 기대 실점 합이 가장 낮은 비율을 계산해요 "
               "(각 구종의 한계 가치가 같아지는 지점 = 게임이론의 균형 조건).")
    k1, k2, k3 = st.columns(3)
    best_k = min(ev, key=ev.get)
    for col_, (k, v) in zip((k1, k2, k3), ev.items()):
        col_.metric(k, f"{v:+.2f}", help="이 가정(β) 아래 100구당 상대 득점 기대 (음수일수록 투수에게 좋음)",
                    delta="가장 좋음" if k == best_k else None, delta_color="off")
    df = df.sort_values("usage", ascending=False)
    dd = pd.DataFrame({
        "구종": [ko(t) for t in df["pitch_type"]],
        "평소 비율": df["usage"] * 100, "추천 비율": df["mix"] * 100,
        "차이(%p)": (df["mix"] - df["usage"]) * 100,
        "기대 실점": df["rv0"], "노릴 코스": [R.describe(x, z) for x, z in zip(df["tx"], df["tz"])],
    })
    g1, g2 = st.columns([3, 2])
    with g1:
        st.dataframe(dd, hide_index=True, width="stretch", column_config={
            "평소 비율": st.column_config.NumberColumn(format="%.0f%%"),
            "추천 비율": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "차이(%p)": st.column_config.NumberColumn(format="%+.0f"),
            "기대 실점": st.column_config.NumberColumn(format="%+.2f",
                                                       help="그 구종을 노릴 만한 코스(상위 25%)에 던졌을 때 100구당 상대 득점. 낮을수록 좋음")})
    with g2:
        fig, ax = plt.subplots(figsize=(4.4, 0.5 * len(df) + 1))
        y = np.arange(len(df))[::-1]
        ax.barh(y + 0.18, df["usage"] * 100, height=0.36, color="#bdc1c6", label="평소")
        ax.barh(y - 0.18, df["mix"] * 100, height=0.36, color=[COLOR.get(t, "#1a73e8") for t in df["pitch_type"]], label="추천")
        ax.set_yticks(y, [name_ko(t) for t in df["pitch_type"]], fontsize=9)
        ax.set_xlabel("비율 (%)", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    top = dd.iloc[dd["차이(%p)"].abs().argmax()]
    st.info(f"💡 이 상황에선 {top['구종']}을(를) 평소보다 {top['차이(%p)']:+.0f}%p — "
            f"{'더 던질 만해요 (기대 실점이 다른 공보다 낮음)' if top['차이(%p)'] > 0 else '줄이는 게 좋아요 (기대 실점이 상대적으로 높음)'}. "
            f"β를 0으로 내리면 '제일 좋은 공만'이 되고, 올리면 평소 비율에 가까워져요.")

    st.subheader("✅ 근거와 한계")
    if gexp:
        t = []
        for yr in (2024, 2025):
            b = gexp[yr]["beta"]
            t.append({"시즌": yr, "스윙 (%p)": f"{b['r_sw'][0] * 10:+.2f} [{b['r_sw'][1][0] * 10:+.2f}, {b['r_sw'][1][1] * 10:+.2f}]",
                      "헛스윙 (%p, 스윙했을 때)": f"{b['r_wh'][0] * 10:+.2f} [{b['r_wh'][1][0] * 10:+.2f}, {b['r_wh'][1][1] * 10:+.2f}]",
                      "실점 가치 (100구당)": f"{b['resid'][0] / 10:+.2f} [{b['resid'][1][0] / 10:+.2f}, {b['resid'][1][1] / 10:+.2f}]"})
        st.markdown("**① 타자는 정말 노리고 들어오나?** — 한 카운트에서 어떤 구종을 평소보다 10%p 더 던질 때, 그 구종에 대한 반응이 "
                    "모델 예상보다 얼마나 달라지나 (같은 투수·같은 구종끼리 비교, 투수 단위 95% 구간)")
        st.dataframe(pd.DataFrame(t), hide_index=True, width="stretch")
        st.markdown("""
- **헛스윙은 두 시즌 모두 줄어요** → 그 카운트에 그 공이 올 걸 알면 타자가 더 잘 맞혀요. 노리고 들어온다는 증거예요.
  (투수는 원래 잘 통하는 공을 더 던지기 때문에 이 효과를 오히려 작게 보이게 만드는데도 나타났어요.)
- 반대로 **실점 가치는 '더 던질수록 좋아짐'으로** 나와요. 이건 노림의 반대가 아니라, 투수가 그 카운트에서 잘 통하는 공을
  골라 더 던지는 **선택 효과**가 섞여서예요. 이 데이터만으로는 둘을 깔끔하게 분리할 수 없어요.
""")
    if gmix:
        st.markdown(f"""
**② β는 어떻게 정했나 (역 게임이론)** — 'MLB 투수들이 실제로 카운트마다 고르는 구종 비율'을 가장 잘 설명하는 β를 2024에서 찾았어요
(투수×카운트 약 4,200칸). β = **{gmix['beta']:g}** 일 때 가장 잘 맞았고, '제일 좋은 공만' 모델보다 훨씬 실제에 가까웠어요
(교차 엔트로피 {gmix['loss24'][gmix['beta']]:.2f} vs {gmix['greedy24']:.2f}, 낮을수록 비슷). 즉 MLB 투수들은 **타자가 꽤 강하게 노린다고 보고** 섞어 던져요.
다만 '모든 카운트에서 평소 비율 그대로'({gmix['same24']:.2f})가 조금 더 잘 맞아서, 카운트별 미세 조정까지 이 모델이 다 설명하진 못해요.

**③ 한계** — 이 추천이 실제로 실점을 줄이는지는 검증하지 못했어요 (투수가 일부러 다르게 던진 실험 데이터가 없어서).
'이런 가정이면 이 비율'을 보여주는 의사결정 도구로 써 주세요. 코스 추천(🛡️ 수비팀)은 2025 데이터로 검증된 부분이에요.
""")
    minimax_section(pid, pname)


# ---------------------------------------------------------------- 🤖 AI 분석관
AIM = "🤖 AI 분석관"
AI_MODELS = {"claude-sonnet-5-5": "Sonnet 5.5 (기본, 빠르고 저렴)", "claude-opus-5-5": "Opus 5.5 (더 깊게, 비쌈)",
             "claude-haiku-5-5": "Haiku 5.5 (가장 빠르고 저렴)"}
AI_EXAMPLES = [
    "스킨스가 저지를 2사 1,2루 풀카운트에서 상대하면 뭘 어디에 던져야 해?",
    "오타니가 스킨스 상대로 1-1 카운트에서 노려야 할 공은?",
    "헤이더가 제일 좋은 공을 더 좋게 만들려면 뭘 바꿔야 해?",
    "콴은 스윙을 바꾸는 게 좋을까?",
    "저지 스카우팅 리포트를 써줘 (투수 입장에서 공략법 포함)",
    "이정후랑 가장 비슷한 MLB 타자는 누구야?",
    "스킨스가 소토 상대로 0-2에서 구종을 어떤 비율로 섞어야 해?",
    "MLB 투수들은 미니맥스대로 던져? 스킨스는?",
]


def ai_key():
    """API 키: .streamlit/secrets.toml → 환경변수 → 이번 접속에서 입력한 값 (저장 안 함)"""
    try:
        k = st.secrets.get("ANTHROPIC_API_KEY")
        if k:
            return k, "secrets.toml"
    except Exception:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"], "환경변수"
    if st.session_state.get("ai_key_input"):
        return st.session_state["ai_key_input"], "이번 접속에서 입력"
    return None, None


@st.cache_resource
def ai_analyst(_t):
    rep, samp, sm = load_mech()
    return AI.Analyst(M, S, D["O"], D["P"], report=rep, samples=samp, swing_v2=sm,
                      move=load_move(), BS=load_bat(), EMB=load_emb())


def ai_tool_label(name, inp):
    nm = {"find_player": "선수 찾기", "pitcher_report": "투수 리포트", "batter_report": "타자 리포트",
          "pitch_recommendation": "볼배합 시뮬레이션", "hitting_plan": "타격 계획", "pitch_shape_targets": "공 모양 목표",
          "batter_swing_targets": "스윙 목표", "validation_notes": "검증 결과",
          "similar_players": "비슷한 선수 찾기", "pitch_mix": "게임이론 구종 비율", "minimax_check": "미니맥스 검증"}.get(name, name)
    return f"🔧 {nm} " + ", ".join(f"{k}={v}" for k, v in (inp or {}).items())


def ai_page():
    with st.sidebar:
        st.header("🤖 AI 분석관")
        model = st.selectbox("언어 모델", list(AI_MODELS), format_func=lambda k: AI_MODELS[k])
        key, src = ai_key()
        if key:
            st.caption(f"API 키: ✅ ({src})")
        else:
            st.text_input("Anthropic API 키", type="password", key="ai_key_input",
                          help="이번 접속에서만 쓰고 저장하지 않아요. 매번 넣기 싫으면 .streamlit/secrets.toml 에 저장")
        if st.button("🗑️ 대화 지우기"):
            st.session_state["ai_msgs"], st.session_state["ai_view"] = [], []
    st.markdown("## 🤖 AI 분석관")
    st.caption("질문하면 AI(Claude)가 이 앱의 모델들(볼배합·타격 계획·공 모양·스윙 목표·검증 결과)을 직접 불러서 숫자를 받고, "
               "그 숫자로만 답해요. 어떤 도구를 썼는지는 답변 아래 '🔧 사용한 도구'에서 확인할 수 있어요.")
    try:
        import anthropic
    except ImportError:
        st.error("anthropic 패키지가 없어요. 터미널에서 `python -m pip install anthropic` 실행 후 앱을 다시 켜세요 "
                 "(run_app.bat 을 다시 실행해도 설치돼요).")
        return
    key, _ = ai_key()
    if not key:
        st.info("""
**처음 한 번만 설정하면 돼요**

1. https://console.anthropic.com 에 가입·로그인 → **Billing** 에서 크레딧 충전 (최소 금액이면 충분)
2. **API Keys → Create Key** → `sk-ant-...` 로 시작하는 키 복사
3. 야구 폴더 안에 `.streamlit` 폴더를 만들고, 그 안에 `secrets.toml` 파일을 만들어 아래 한 줄 저장
   ```
   ANTHROPIC_API_KEY = "sk-ant-여기에_키"
   ```
4. 앱 새로고침. (급하면 왼쪽 칸에 키를 붙여 넣어도 돼요 — 이번 접속에서만 쓰고 저장 안 함)

⚠️ 키는 비밀번호와 같아요. 카톡·깃허브 등에 올리지 마세요.
""")
        return
    st.session_state.setdefault("ai_msgs", [])
    st.session_state.setdefault("ai_view", [])
    if not st.session_state["ai_view"]:
        st.markdown("**이런 걸 물어볼 수 있어요**")
        cols = st.columns(2)
        for i, q in enumerate(AI_EXAMPLES):
            if cols[i % 2].button(q, key=f"ai_ex{i}", width="stretch"):
                st.session_state["ai_pending"] = q
                st.rerun()
    for v in st.session_state["ai_view"]:
        with st.chat_message(v["role"]):
            st.markdown(v["text"])
            if v.get("log"):
                with st.expander(f"🔧 사용한 도구 {len(v['log'])}개"):
                    for l in v["log"]:
                        st.markdown(f"**{ai_tool_label(l['tool'], l['input'])}**")
                        st.json(l["output"], expanded=False)
    q = st.chat_input("예: 스킨스가 저지한테 0-2에서 뭘 던져야 해?") or st.session_state.pop("ai_pending", None)
    if not q:
        return
    st.session_state["ai_view"].append({"role": "user", "text": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        status = st.status("생각하는 중...", expanded=True)
        try:
            client = anthropic.Anthropic(api_key=key)
            msgs = st.session_state["ai_msgs"] + [{"role": "user", "content": q}]
            text, log, msgs = AI.chat(client, ai_analyst(D["t"]), msgs, model=model,
                                      on_tool=lambda n, i: status.write(ai_tool_label(n, i)))
            status.update(label=f"완료 — 도구 {len(log)}개 사용", state="complete", expanded=False)
        except Exception as e:
            status.update(label="오류", state="error")
            msg = str(e)
            if "authentication" in msg.lower() or "401" in msg:
                msg = "API 키가 올바르지 않아요. 키를 다시 확인해 주세요."
            elif "credit" in msg.lower() or "billing" in msg.lower():
                msg = "API 크레딧이 부족해요. Anthropic 콘솔의 Billing에서 충전해 주세요."
            st.error(msg)
            st.session_state["ai_view"].pop()
            return
        st.markdown(text)
        if log:
            with st.expander(f"🔧 사용한 도구 {len(log)}개"):
                for l in log:
                    st.markdown(f"**{ai_tool_label(l['tool'], l['input'])}**")
                    st.json(l["output"], expanded=False)
    st.session_state["ai_msgs"] = msgs
    st.session_state["ai_view"].append({"role": "assistant", "text": text, "log": log})


# ---------------------------------------------------------------- 📋 스카우팅 리포트
REPM = "📋 스카우팅 리포트"


@st.cache_resource
def load_teams():
    import scouting as SC
    return SC.load_teams(M)


def rep_page():
    import scouting as SC
    T = load_teams()
    teams = sorted(T)
    pn = dict(zip(ars_all["pitcher"], ars_all["player_name"]))
    with st.sidebar:
        st.header("📋 시리즈 스카우팅 리포트")
        opp = st.selectbox("상대 팀 (타선)", teams, index=teams.index("NYY") if "NYY" in teams else 0, key="r_opp")
        us = st.selectbox("우리 팀 (선발)", teams, index=teams.index("PIT") if "PIT" in teams else 0, key="r_us")
        default = [pn[p] for p in T[us]["starters"] if p in pn]
        picks = st.multiselect("선발 투수 (바꿔도 됨)", sorted(pitcher_names), default=default, max_selections=4, key="r_p")
    lineup = T[opp]["lineup"]
    st.markdown(f"## 📋 {us} 투수진 vs {opp} 타선")
    st.caption("상대 타선 = 2024 시즌 가장 자주 나온 선발 라인업. 타자마다 카운트(초구·유리·불리·풀카운트) × 우리 선발별로 "
               "게임이론 구종 비율과 구종별 목표 코스, 수비 위치를 한 페이지씩 정리해요. 인쇄하거나 PDF로 저장할 수 있어요.")
    st.dataframe(pd.DataFrame([{"타순": s, "타자": SC.flipname(M["batter_names"].get(b, b)), "선발 경기": g} for s, b, g in lineup]),
                 hide_index=True, width="stretch")
    key = (opp, us, tuple(picks))
    if st.button(f"리포트 만들기 (타자 {len(lineup)}명 × 투수 {len(picks)}명, 1~2분)", type="primary", disabled=not picks):
        bar = st.progress(0.0, text="계산 중...")
        done = []

        def log(msg):
            done.append(msg)
            bar.progress(min(len(done) / max(len(lineup), 1), 1.0), text=msg)
        rep = SC.build(M, S, D["B"], load_bat(), load_emb(), opp, us, lineup, [pitcher_names[n] for n in picks], log=log)
        st.session_state["rep_html"] = (key, SC.render(rep))
        bar.empty()
    got = st.session_state.get("rep_html")
    if got and got[0] == key:
        st.download_button("⬇️ 리포트 내려받기 (HTML · 브라우저에서 인쇄 → PDF 저장)", got[1].encode("utf-8"),
                           file_name=f"scouting_{us}_vs_{opp}.html", mime="text/html")
        if hasattr(st, "iframe"):          # Streamlit 1.5x+ (리포트는 이 코드가 만든 HTML이라 그대로 보여 줘도 됨)
            st.iframe(got[1], height=900)
        else:
            import streamlit.components.v1 as components
            components.html(got[1], height=900, scrolling=True)
    st.info("근거: 코스 추천은 2025 백테스트로 검증됐고(추천 코스 근처로 간 공 100구당 −2.6점), 구종 '순위'는 검증되지 않아 "
            "게임이론 비율로만 보여 줘요. 스윙·헛스윙 확률은 LightGBM + 선수 임베딩 신경망 평균이에요.")


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    mode = st.radio("모드", [AIM, "🛡️ 수비팀", "⚔️ 공격팀", GAMEM, "🎯 투구 궤적", MECH, BAT, EMBM, REPM], index=1, horizontal=True,
                    label_visibility="collapsed")

if mode == MECH:
    mech_page()
    st.stop()
if mode == BAT:
    bat_page()
    st.stop()
if mode == AIM:
    ai_page()
    st.stop()
if mode == EMBM:
    emb_page()
    st.stop()
if mode == GAMEM:
    game_page()
    st.stop()
if mode == REPM:
    rep_page()
    st.stop()

with st.sidebar:
    st.header("⚾ 매치업 설정")
    pname = st.selectbox("투수", list(pitcher_names), index=list(pitcher_names).index("Skenes, Paul")
                         if "Skenes, Paul" in pitcher_names else 0)
    bname = st.selectbox("타자", batter_list, index=batter_list.index("Judge, Aaron")
                         if "Judge, Aaron" in batter_list else 0)
    pid, bid = pitcher_names[pname], batters[bname]
    auto_stand = M["batter_stand"].get(bid, "R")
    stand = st.radio("타석", ["R", "L"], index=0 if auto_stand == "R" else 1, horizontal=True,
                     format_func=lambda s: "우타" if s == "R" else "좌타",
                     help="스위치 타자면 실제로 들어선 타석을 고르세요")
    my_types = ars_all[ars_all["pitcher"] == pid].sort_values("usage", ascending=False)["pitch_type"].tolist()

    if mode != "🎯 투구 궤적":
        st.subheader("카운트 · 상황")
        c1, c2, c3 = st.columns(3)
        balls = c1.selectbox("볼", [0, 1, 2, 3])
        strikes = c2.selectbox("스트", [0, 1, 2])
        outs = c3.selectbox("아웃", [0, 1, 2])
        b1, b2, b3 = st.columns(3)
        r1, r2, r3 = b1.checkbox("1루"), b2.checkbox("2루"), b3.checkbox("3루")
        sit = (outs, int(r1), int(r2), int(r3))

        st.subheader("직전 공")
        prev_type = st.selectbox("구종", ["모름 / 초구"] + my_types,
                                 format_func=lambda t: t if t.startswith("모름") else ko(t))
        prev_key = None
        if not prev_type.startswith("모름"):
            p1, p2 = st.columns(2)
            side = p1.selectbox("좌우", ["바깥쪽", "가운데", "몸쪽"])
            height = p2.selectbox("높이", ["낮게", "가운데", "높게"])
            prev_key = (prev_type, SIDE[side], HEIGHT[height])

    if mode == "🛡️ 수비팀":
        st.subheader("투수 제구력")
        command = st.slider("제구 오차 (피트)", 0.3, 0.8, 0.5, 0.05,
                            help="작을수록 제구가 좋은 투수. 0.5가 리그 평균 정도")
    if mode == "⚔️ 공격팀":
        st.subheader("상대 수비 배치")
        if_al = st.selectbox("내야", [0, 1, 2], format_func=IF_KO.get,
                             index=1 if stand == "L" else 0,
                             help="2023년부터 극단적 시프트(2루 한쪽에 내야수 3명)는 금지. 이 세 가지만 가능해요")
        of_al = st.selectbox("외야", [0, 1], format_func=OF_KO.get)

prev_txt = ""
if mode != "🎯 투구 궤적" and prev_key:
    prev_txt = f" · 직전 공 {ko(prev_key[0])}"

# =================================================================== 🛡️ 수비팀
if mode == "🛡️ 수비팀":
    with st.spinner("볼배합 계산 중..."):
        res = run_defense(pid, bid, balls, strikes, stand, command, prev_key, sit, D["t"])
    best = res.loc[res.groupby("pitch_type")["rv100"].idxmin()].sort_values("rv100")
    # 피할 곳 = 타자가 '칠 수 있는' 구역(존 + 바로 바깥)에서 가장 위험한 곳 = 실투 주의 구역
    hit = res[res["tx"].between(-0.95, 0.95) & res["tz"].between(-0.1, 1.1)]
    worst = hit.loc[hit.groupby("pitch_type")["rv100"].idxmax()].set_index("pitch_type")
    near = res[res["tx"].between(-1.1, 1.1) & res["tz"].between(-0.3, 1.2)]
    w = near.groupby("pitch_type").agg(rv=("rv100", "mean"), usage=("usage", "first"))
    baseline = (w["rv"] * w["usage"]).sum() / w["usage"].sum()
    top = best.iloc[0]

    header("🛡️ 수비팀: 어떻게 막을까", pname, bname, stand, balls, strikes, outs, sit, prev_txt)
    m1, m2, m3 = st.columns(3)
    m1.metric("추천", f"{name_ko(top['pitch_type'])} · {top['velo']:.0f}mph")
    m1.markdown(f"📍 **{R.describe(top['tx'], top['tz'])}**")
    m2.metric("리그 평균 타자와 비교하면", verdict(top["rv100"]))
    m2.caption(f"추천 공 100개 → 상대 득점 {top['rv100']:+.1f}점 · 평소처럼 던지면 {baseline:+.1f}점")
    m3.metric("평소보다 덜 내주는 점수", f"{baseline - top['rv100']:.1f}점")
    m3.caption("선발 한 경기(약 100구) 내내 이렇게 던졌을 때 기준")

    if top["rv100"] >= 1:
        st.info(f"💡 **{bname}**은(는) 어떤 공을 던져도 리그 평균 타자보다 위험한 상대예요 (최선의 공도 {top['rv100']:+.1f}점). "
                f"'타자 유리'는 리그 평균 타자와 비교한 결과라서, 이럴 땐 **가장 덜 위험한 1순위**를 던지면 돼요. "
                f"평소처럼 던질 때보다 100구당 {baseline - top['rv100']:.1f}점을 덜 내줘요.")
    t1, t2, t3, t4 = st.tabs(["🎯 볼배합", "🧤 수비 위치", "🧢 타자 분석", "💪 투수 구종"])
    with t1:
        top_z, bot_z = zone_h(bid)
        flip = -1 if stand == "L" else 1
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            fig, ax = plt.subplots(figsize=(4.4, 5.6))
            zone_axes(ax, stand, top_z - bot_z, "투수 시점 · 번호 = 추천 순위")
            wst = worst.loc[top["pitch_type"]]
            x_mark(ax, wst["tx"] * flip, wst["tz"], "실투 주의")
            markers(ax, best.head(3), flip)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right:
            st.markdown("##### 구종별 최적 코스")
            html = ""
            for i, (_, r) in enumerate(best.iterrows(), 1):
                wst = worst.loc[r["pitch_type"]]
                sw = f"스윙 {r['p_swing']:.0%} · 헛스윙 {r['p_whiff']:.0%} · " if S is not None else ""
                html += card(COLOR.get(r["pitch_type"], "#888"), f"{i}. {name_ko(r['pitch_type'])}",
                             f"{r['pitch_type']} · {r['velo']:.0f}mph", rel_label(r["rv100"], baseline),
                             f"📍 {R.describe(r['tx'], r['tz'])} <span style='opacity:.6'>({r['rv100']:+.1f}점/100구)</span>"
                             f"<br><span style='font-size:.85rem'>💬 {reason(r)}</span>",
                             f"{sw}실투 주의: {R.describe(wst['tx'], wst['tz'])} ({wst['rv100']:+.1f})")
            st.markdown(html, unsafe_allow_html=True)
            st.caption("✅/👍/➖/⚠️ = 이 타자에게 평소처럼 던질 때와 비교한 추천 정도. "
                       "스윙률·헛스윙률은 그 코스를 **노렸을 때**(제구 오차 포함) 기준이라, 존 밖을 노려도 일부는 존 안으로 들어가요. "
                       "실투 주의 = 타자가 칠 수 있는 구역 중 가장 많이 얻어맞는 곳.")
            st.caption("📌 2025 시즌 백테스트에서 **코스 추천**은 효과가 확인됐고, **구종 순위**는 확인되지 않았어요. "
                       "구종은 섞어 던지는 걸 전제로 참고만 하세요.")
        with st.expander("🗺️ 구종별 코스 히트맵 (전체)"):
            fig = R.heatmap(res, f"{pname} vs {bname}  {balls}-{strikes}", None, stand, korean=True)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    with t2:
        Bm = D["B"]
        if Bm is None:
            st.warning("bip_model.pkl이 없어요. build_bip.py를 먼저 실행하세요.")
        else:
            bip, fallback = ST.batter_bip(Bm, bid, stand)
            lg_bip = Bm["batter_bip"][Bm["batter_bip"]["stand"] == stand]
            summ, lg = ST.spray_summary(bip), ST.spray_summary(lg_bip)
            if fallback:
                st.info("이 타자는 타구 표본이 적어서 같은 타석 방향 리그 평균으로 보여 줘요.")
            st.markdown("##### 이 타자의 타구는 어디로 가나")
            fig = spray_fan(summ, lg, stand)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            gb_pull, lg_gb_pull = summ["땅볼"]["당김"], lg["땅볼"]["당김"]
            air_pull, lg_air_pull = summ["공중 타구"]["당김"], lg["공중 타구"]["당김"]
            tips = []
            if gb_pull >= lg_gb_pull + 0.05:
                tips.append(f"**내야**: 땅볼의 {gb_pull:.0%}가 당긴 쪽 (리그 {lg_gb_pull:.0%}) → "
                            f"**{'3루' if stand == 'R' else '1루'} 쪽으로 쉐이드**")
            elif gb_pull <= lg_gb_pull - 0.05:
                tips.append(f"**내야**: 땅볼을 덜 당김 ({gb_pull:.0%}, 리그 {lg_gb_pull:.0%}) → **정상 수비, 반대쪽 간격 좁히기**")
            else:
                tips.append(f"**내야**: 땅볼 방향이 리그 평균과 비슷 ({gb_pull:.0%}) → **정상 수비**")
            if air_pull >= lg_air_pull + 0.05:
                tips.append(f"**외야**: 공중 타구의 {air_pull:.0%}가 당긴 쪽 (리그 {lg_air_pull:.0%}) → "
                            f"**{'좌익수·중견수' if stand == 'R' else '우익수·중견수'}를 당긴 쪽으로**")
            else:
                tips.append(f"**외야**: 공중 타구 방향이 고른 편 ({air_pull:.0%} 당김) → **외야 정상**")
            loc = M["profile"]["loc"]
            lp = loc[loc["batter"] == bid].set_index("xloc")["b_pull_loc"]
            dl = M["profile"]["defaults"]["b_pull_loc"]
            pl = {k: lp.get(k, dl[k]) for k in (0, 1, 2)}
            tips.append(f"**투구와 연결**: 이 타자는 몸쪽 공을 {pl[2]:.0%}, 바깥쪽 공을 {pl[0]:.0%} 당겨요 → "
                        f"당긴 쪽으로 수비를 옮겼다면 **몸쪽 공**으로 타구를 수비 쪽에 보내고, "
                        f"정상 수비라면 바깥쪽 공으로 당기기 어렵게")
            gp = M["profile"]["group"]
            gbg = gp[gp["batter"] == bid].set_index("group")["b_gb_g"] if "b_gb_g" in gp else pd.Series()
            if len(gbg):
                g_best = gbg.idxmax()
                tips.append(f"**땅볼 유도**: {GROUP_KO[g_best]}에 맞으면 땅볼 {gbg.max():.0%} → 내야 수비를 믿는다면 {GROUP_KO[g_best]} 낮게")
            st.markdown("##### 추천 수비 위치")
            for t in tips:
                st.markdown("- " + t)

            with st.expander("📊 수비 배치별 예상 효과 (모델 계산)"):
                av = ST.align_values(Bm, bip)
                av["배치"] = av["if_al"].map(IF_KO) + " + " + av["of_al"].map(OF_KO)
                av["예상 wOBA (맞았을 때)"] = av["woba"].round(3)
                av["타구 100개당 실점 (정상 대비)"] = av["runs100"].round(1)
                st.dataframe(av[["배치", "예상 wOBA (맞았을 때)", "타구 100개당 실점 (정상 대비)"]],
                             hide_index=True, width="stretch")
                st.caption("이 타자의 실제 타구들을 각 배치에 넣어 본 결과예요. 2024 데이터에서 배치 '이름'의 효과는 아주 작게 나와요 "
                           "(극단적 시프트 금지 후, '정상 수비'로 기록돼도 수비수들이 이미 타자별로 위치를 조정하기 때문). "
                           "그래서 위의 타구 방향 분포를 보고 위치를 정하는 게 더 확실해요.")

    with t3:
        batter_tab(bid, stand)
    with t4:
        pitcher_tab(pid, pname)

# =================================================================== ⚔️ 공격팀
elif mode == "⚔️ 공격팀":
    if D["O"] is None:
        st.error("offense_model.pkl이 없어요. build_offense.py를 먼저 실행하세요.")
        st.stop()
    with st.spinner("노림수 계산 중..."):
        off = run_offense(pid, bid, balls, strikes, stand, prev_key, sit, (if_al, of_al), D["t"])
        dfn = run_defense(pid, bid, balls, strikes, stand, 0.5, prev_key, sit, D["t"])
    zone = off[off["tx"].between(-0.85, 0.85) & off["tz"].between(0, 1)]
    hunt = zone.loc[zone.groupby("pitch_type")["value"].idxmax()].sort_values("value", ascending=False)
    # 참을 공: 존 바로 바깥 '유인구 띠'에서만 고름 (누가 봐도 볼인 공은 제외)
    #  → 이 타자가 실제로 자주 따라 나가는(스윙률 높은) 곳 중에서 치면 손해가 큰 곳
    off = off.merge(dfn[["pitch_type", "tx", "tz", "p_swing"]], on=["pitch_type", "tx", "tz"], how="left")
    chase = off[off["tx"].between(-1.1, 1.1) & off["tz"].between(-0.35, 1.25)
                & ~(off["tx"].between(-0.83, 0.83) & off["tz"].between(0, 1))].copy()
    chase["trap"] = chase["value"] * chase["p_swing"].fillna(0.3)
    lay = chase.loc[chase.groupby("pitch_type")["trap"].idxmin()].set_index("pitch_type")
    pbest = dfn.loc[dfn["rv100"].idxmin()]
    pv = off[(off["pitch_type"] == pbest["pitch_type"]) & (off["tx"] == pbest["tx"]) & (off["tz"] == pbest["tz"])]
    pv = pv["value"].iloc[0] if len(pv) else np.nan

    header("⚔️ 공격팀: 어떻게 칠까", pname, bname, stand, balls, strikes, outs, sit,
           prev_txt + f" · 상대 {IF_KO[if_al]} / {OF_KO[of_al]}")
    h = hunt.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("노릴 공", f"{name_ko(h['pitch_type'])} · {h['velo']:.0f}mph")
    m1.markdown(f"📍 **{R.describe(h['tx'], h['tz'])}**  ·  치면 100구당 **{h['value']:+.1f}점**")
    worst_lay = lay.sort_values("trap").iloc[0]
    m2.metric("속지 말 유인구", f"{name_ko(lay['trap'].idxmin())}")
    m2.markdown(f"📍 **{R.describe(worst_lay['tx'], worst_lay['tz'])}**  ·  스윙률 {worst_lay['p_swing']:.0%}, "
                f"치면 100구당 **{worst_lay['value']:+.1f}점**")
    m3.metric("상대 투수의 최선", f"{name_ko(pbest['pitch_type'])}")
    m3.markdown(f"📍 {R.describe(pbest['tx'], pbest['tz'])} → "
                + ("**참는 게 이득**" if pv < 0 else "**쳐도 됨**") + f" ({pv:+.1f})")

    t1, t2, t3 = st.tabs(["🎯 노림수 지도", "🏏 타구 방향 전략", "🧢 타자 분석"])
    with t1:
        top_z, bot_z = zone_h(bid)
        flip = -1 if stand == "L" else 1
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            fig, ax = plt.subplots(figsize=(4.4, 5.6))
            zone_axes(ax, stand, top_z - bot_z, "투수 시점 · 번호 = 노릴 공 순위")
            x_mark(ax, worst_lay["tx"] * flip, worst_lay["tz"], "참기", color="#7b1fa2")
            markers(ax, hunt.head(3), flip)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right:
            st.markdown("##### 구종별 노릴 코스 / 참을 코스")
            html = ""
            for i, (_, r) in enumerate(hunt.iterrows(), 1):
                l = lay.loc[r["pitch_type"]]
                html += card(COLOR.get(r["pitch_type"], "#888"), f"{i}. {name_ko(r['pitch_type'])}",
                             f"{r['pitch_type']} · {r['velo']:.0f}mph · 구사율 {r['usage']:.0%}",
                             f"🟢 {r['value']:+.1f}",
                             f"🏏 노릴 곳: <b>{R.describe(r['tx'], r['tz'])}</b>",
                             f"✋ 속지 말 곳: {R.describe(l['tx'], l['tz'])} (스윙률 {l['p_swing']:.0%}, 치면 {l['value']:+.1f})")
            st.markdown(html, unsafe_allow_html=True)
            st.caption("숫자 = 그 코스 공에 스윙했을 때와 참았을 때의 차이 (100구당 득점, 타자 기준). "
                       "+면 치는 게 이득, −면 참는 게 이득.")
        with st.expander("🗺️ 구종별 스윙 지도 (전체)"):
            types = off.groupby("pitch_type")["usage"].first().sort_values(ascending=False).index
            vmax = np.abs(off["value"]).quantile(0.95)
            fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 4.2), squeeze=False)
            for ax, t in zip(axes[0], types):
                d = off[off["pitch_type"] == t].pivot(index="tz", columns="tx", values="value")
                if flip == -1:
                    d = d.iloc[:, ::-1]
                im = ax.imshow(d.values, origin="lower", cmap="PRGn", vmin=-vmax, vmax=vmax, aspect="auto",
                               extent=[R.X_TARGETS[0], R.X_TARGETS[-1], R.Z_TARGETS[0], R.Z_TARGETS[-1]])
                ax.add_patch(plt.Rectangle((-0.83, 0), 1.66, 1, fill=False, lw=2, color="black"))
                ax.set_title(f"{name_ko(t)} ({t})", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("몸쪽 ←   → 바깥쪽" if flip == -1 else "바깥쪽 ←   → 몸쪽")
            axes[0][0].set_ylabel("낮게 ←   → 높게")
            fig.suptitle(f"{bname} 스윙 지도 (투수 시점)\n초록 = 쳐라, 보라 = 참아라", fontsize=11, y=1.06)
            fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="스윙 가치 (100구당 득점)")
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    with t2:
        Bm = D["B"]
        if Bm is None:
            st.warning("bip_model.pkl이 없어요. build_bip.py를 먼저 실행하세요.")
        else:
            bip, fallback = ST.batter_bip(Bm, bid, stand)
            dv = ST.direction_values(Bm, bip, if_al, of_al)
            st.markdown(f"##### 상대가 **{IF_KO[if_al]} / {OF_KO[of_al]}**일 때, 어느 방향으로 쳐야 하나")
            if fallback:
                st.info("이 타자는 타구 표본이 적어서 같은 타석 방향 리그 평균 타구로 계산했어요.")
            cols = st.columns(len(dv))
            for col, (_, r) in zip(cols, dv.iterrows()):
                vals = {k: r[k] for k in ("당겨서", "가운데로", "밀어서")}
                best_dir = max(vals, key=vals.get)
                with col:
                    st.markdown(f"**{r['타구']}** <span style='opacity:.6'>(타구의 {r['비율']:.0%})</span>",
                                unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(3.2, 2.4))
                    names = list(vals)
                    bars = ax.bar(names, [vals[k] for k in names],
                                  color=["#1a73e8" if k == best_dir else "#c8ccd2" for k in names])
                    for b_, k in zip(bars, names):
                        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{vals[k]:.3f}",
                                ha="center", va="bottom", fontsize=9)
                    ax.set_ylim(0, max(vals.values()) * 1.25)
                    ax.set_yticks([])
                    for sp in ("top", "right", "left"):
                        ax.spines[sp].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                    st.markdown(f"→ **{best_dir}** 칠 때 가장 좋아요")
            st.caption("숫자 = 그 방향으로 갔을 때 타구 하나의 평균 가치(wOBA: 아웃 0, 단타 약 0.9, 홈런 약 2.0). "
                       "이 타자의 실제 타구 속도·각도는 그대로 두고 방향만 바꿔서 계산했어요.")
            st.info("💡 같은 속도·각도라도 방향에 따라 가치가 크게 달라요. 수비수들이 이미 타자의 당긴 쪽에 서 있기 때문이에요. "
                    "배치 이름(정상/쉐이드/전략적)보다 이 방향 차이가 훨씬 커요.")
    with t3:
        batter_tab(bid, stand)

# =================================================================== 🎯 투구 궤적
else:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:                     # 처음 실행이면 자동으로 설치
        import subprocess
        import sys
        with st.spinner("3D 그래프 도구(plotly)를 처음 한 번 설치하는 중... 1분 정도 걸려요"):
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "plotly"], check=False)
        try:
            import plotly.graph_objects as go
        except ModuleNotFoundError:
            st.error("plotly 설치에 실패했어요. 터미널에서 `pip install plotly`를 실행한 뒤 새로고침해 주세요.")
            st.stop()

    PH = D["P"]
    if PH is None:
        st.error("physics.pkl이 없어요. build_physics.py를 먼저 실행하세요.")
        st.stop()
    ph = PH[PH["pitcher"] == pid].set_index("pitch_type")
    types = [t for t in my_types if t in ph.index] or list(ph.index)
    top_z, bot_z = zone_h(bid)
    zh = top_z - bot_z

    with st.sidebar:
        st.subheader("던질 공")
        fb_first = [t for t in types if t in ("FF", "SI", "FC")] + [t for t in types if t not in ("FF", "SI", "FC")]
        t_a = st.selectbox("구종 A", fb_first, format_func=ko)
        s_a = st.selectbox("좌우 A", list(SIDE), index=1, key="sa")
        h_a = st.selectbox("높이 A", list(HEIGHT), index=1 if t_a in ("FF", "SI", "FC") else 3, key="ha")
        st.subheader("비교할 공 (터널링)")
        others = ["없음"] + [t for t in types if t != t_a]
        default_b = next((t for t in others if t in ("CH", "FS", "FO")), others[1] if len(others) > 1 else "없음")
        t_b = st.selectbox("구종 B", others, index=others.index(default_b),
                           format_func=lambda t: t if t == "없음" else ko(t))
        b_mode = st.radio("B 코스", ["A와 같은 터널로 (자동)", "직접 고르기"], index=0,
                          help="자동: 타자가 결정하는 순간까지 A와 똑같은 길로 오게 던졌을 때 B가 어디에 도착하는지")
        if b_mode == "직접 고르기":
            s_b = st.selectbox("좌우 B", list(SIDE), index=1, key="sb")
            h_b = st.selectbox("높이 B", list(HEIGHT), index=3, key="hb")
        with st.expander("🔧 릴리스 포인트 바꿔 보기"):
            d_h = st.slider("타점 높이 (피트)", -1.5, 1.5, 0.0, 0.1, help="+면 더 높은 곳에서 놓음")
            d_s = st.slider("좌우 (피트)", -1.0, 1.0, 0.0, 0.1, help="+면 몸 바깥쪽으로 벌림 (사이드암 쪽)")
            d_e = st.slider("익스텐션 (피트)", -1.0, 1.0, 0.0, 0.1, help="+면 홈에 더 가까이 끌고 나와서 놓음")

    def target(side, height):
        x_in, zn = SIDE[side], HEIGHT[height]
        return (x_in if stand == "L" else -x_in), bot_z + zn * zh     # 타자 기준 → 스탯캐스트 좌표

    changed = any(abs(v) > 1e-9 for v in (d_h, d_s, d_e))
    pa = T.with_release(ph.loc[t_a].to_dict(), d_h, d_s, d_e)
    tr_a = T.path(pa, *target(s_a, h_a))
    trs = [(f"A {name_ko(t_a)}", COLOR.get(t_a, "#d22d49"), tr_a, t_a)]
    pb = tr_b = None
    if t_b != "없음":
        pb = T.with_release(ph.loc[t_b].to_dict(), d_h, d_s, d_e)
        tgt_b = T.same_tunnel_target(pb, tr_a) if b_mode.startswith("A와") else target(s_b, h_b)
        tr_b = T.path(pb, *tgt_b)
        trs.append((f"B {name_ko(t_b)}", COLOR.get(t_b, "#1a73e8"), tr_b, t_b))

    st.title("🎯 투구 궤적")
    st.markdown(f"#### {pname} &nbsp;→&nbsp; {bname} ({'우타' if stand == 'R' else '좌타'})")
    if changed:
        st.warning(f"🔧 릴리스를 바꿔 본 상태예요: 타점 {d_h:+.1f}피트 · 좌우 {d_s:+.1f}피트 · 익스텐션 {d_e:+.1f}피트")

    # ---- 릴리스 포인트 (타점) 카드
    hand = ph.loc[t_a, "p_throws"]
    lg = PH[(PH["pitch_type"] == t_a) & (PH["p_throws"] == hand)]
    lg_h, lg_side, lg_ext = lg["release_pos_z"].mean(), lg["release_pos_x"].abs().mean(), (60.5 - lg["release_pos_y"]).mean()
    rel_h, rel_side, rel_ext = pa["release_pos_z"], abs(pa["release_pos_x"]), 60.5 - pa["release_pos_y"]
    vaa_a, haa_a = T.approach_angles(tr_a)
    st.markdown("##### 🤚 릴리스 포인트 (공을 놓는 위치)")
    r1c, r2c, r3c, r4c = st.columns(4)
    r1c.metric("타점 높이", f"{rel_h:.2f}피트")
    r1c.caption(f"{rel_h * 30.48:.0f}cm · 리그 평균 대비 {rel_h - lg_h:+.2f}피트")
    r2c.metric("몸에서 옆으로", f"{rel_side:.2f}피트")
    r2c.caption(f"리그 평균 대비 {rel_side - lg_side:+.2f}피트 (+면 더 옆으로)")
    r3c.metric("익스텐션", f"{rel_ext:.2f}피트",
               help="투수판에서 얼마나 앞으로 나와서 공을 놓나. 길수록 타자가 느끼는 공이 빨라요")
    r3c.caption(f"리그 평균 대비 {rel_ext - lg_ext:+.2f}피트")
    r4c.metric("홈 진입 각도 (VAA)", f"{vaa_a:.1f}°",
               help="0°에 가까울수록 평평하게 들어옴. 높은 직구는 평평할수록 떠오르는 듯 보여 헛스윙이 많아요")
    lg_vaa = np.mean([T.approach_angles(T.path(r, *target(s_a, h_a)))[0]
                      for r in lg.sample(min(40, len(lg)), random_state=0).to_dict("records")])
    flat = vaa_a - lg_vaa
    st.caption(f"같은 코스({s_a}, {h_a})에 리그 평균 {name_ko(t_a)}을(를) 던지면 진입 각도 {lg_vaa:.1f}° → 이 투수는 "
               + (f"{flat:+.1f}° 더 평평하게 들어와요." if flat > 0 else f"{-flat:.1f}° 더 가파르게 들어와요.")
               + (" 타점이 낮거나 익스텐션이 길수록 평평해져요." if t_a in ("FF", "SI", "FC") else ""))

    # ---- 신체조건 (player_bio.csv: fetch_bio.py로 MLB Stats API에서 받음)
    bio = D["bio"]
    height_ft = 6.25
    if bio is not None and pid in bio.index and not pd.isna(bio.loc[pid, "height_in"]):
        h_in, w_lb = bio.loc[pid, "height_in"], bio.loc[pid, "weight_lb"]
        height_ft = h_in / 12
        lg_bio = bio[bio.index.isin(PH["pitcher"].unique())]
        rz_h, lg_rz_h = rel_h / height_ft, (PH.groupby("pitcher")["release_pos_z"].mean()
                                           / (bio["height_in"] / 12)).dropna().median()
        ex_h = rel_ext / height_ft
        st.markdown("##### 🧍 신체조건")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("키", f"{h_in // 12:.0f}'{h_in % 12:.0f}\"  ({h_in * 2.54:.0f}cm)")
        b1.caption(f"MLB 투수 평균 {lg_bio['height_in'].mean() * 2.54:.0f}cm")
        b2.metric("몸무게", f"{w_lb:.0f}lb  ({w_lb * 0.4536:.0f}kg)")
        b2.caption(f"MLB 투수 평균 {lg_bio['weight_lb'].mean() * 0.4536:.0f}kg")
        b3.metric("키 대비 타점", f"{rz_h:.2f}", help="타점 높이 ÷ 키. 1.0이면 정수리 높이에서 놓음")
        b3.caption(f"평균 {lg_rz_h:.2f} → " + ("키에 비해 **낮게** 놓아요 (공이 더 평평하게 들어옴)" if rz_h < lg_rz_h - 0.03
                                            else "키에 비해 **높게** 놓아요 (위에서 내리꽂는 편)" if rz_h > lg_rz_h + 0.03
                                            else "키에 맞는 보통 높이예요"))
        b4.metric("키 대비 익스텐션", f"{ex_h:.2f}", help="익스텐션 ÷ 키. 클수록 몸을 길게 끌고 나와서 놓음")
        b4.caption(f"{rel_ext:.1f}피트 = 키의 {ex_h:.0%}")
        st.caption("2024 데이터로 확인해 보니 키 자체는 헛스윙과 관계가 거의 없고(작은 투수 22% vs 큰 투수 24%), "
                   "**키 대비 타점이 낮을수록** 높은 포심 헛스윙이 많아요(26% vs 20%). 다만 이건 진입 각도(VAA)와 같은 효과라, "
                   "모델에 따로 넣어도 예측이 좋아지지 않아서 화면 설명과 투구 동작 크기에만 써요.")

    # ---- 릴리스가 헛스윙에 주는 영향 (스윙/헛스윙 모델)
    if S is not None and "vaa_rel" in S["features"] and "vaa_rel" in ars_all:
        orig = ph.loc[t_a].to_dict()
        vaa_orig = T.approach_angles(T.path(orig, *target(s_a, h_a)))[0]
        dv = vaa_a - vaa_orig                              # 릴리스를 바꿔서 생긴 진입각 변화
        tx, tz = SIDE[s_a], HEIGHT[h_a]
        sw0, wh0 = whatif_swing(pid, bid, stand, t_a, 0.0, 0.0, 0.0, 0.0, tx, tz, D["t"])
        st.markdown(f"##### 🎯 모델이 본 효과: {name_ko(t_a)} {s_a}·{h_a} vs {bname} (0-0, 정확히 그 코스로)")
        w1, w2, w3 = st.columns(3)
        if changed:
            sw1, wh1 = whatif_swing(pid, bid, stand, t_a, d_h, d_s, d_e, dv, tx, tz, D["t"])
            w1.metric("헛스윙률 (스윙했을 때)", f"{wh1:.0%}", f"{(wh1 - wh0) * 100:+.0f}%p", delta_color="normal")
            w2.metric("스윙률", f"{sw1:.0%}", f"{(sw1 - sw0) * 100:+.0f}%p", delta_color="off")
            w3.metric("진입 각도 변화", f"{dv:+.1f}°", help="+면 더 평평해짐")
            arow = ars_all[(ars_all["pitcher"] == pid) & (ars_all["pitch_type"] == t_a)].iloc[0]
            lga = ars_all[ars_all["pitch_type"] == t_a]
            new_vals = {"타점 높이": arow["rel_z"] + d_h, "익스텐션": arow["ext"] + d_e,
                        "몸에서 옆으로": arow["rel_side"] + d_s, "진입각(같은 높이 대비)": arow["vaa_rel"] + dv}
            cols_map = {"타점 높이": "rel_z", "익스텐션": "ext", "몸에서 옆으로": "rel_side", "진입각(같은 높이 대비)": "vaa_rel"}
            out = [k for k, v in new_vals.items()
                   if not (lga[cols_map[k]].quantile(0.02) <= v <= lga[cols_map[k]].quantile(0.98))]
            if out:
                st.warning(f"⚠️ 바꾼 {', '.join(out)}이(가) 실제 MLB {name_ko(t_a)} 투수들 범위를 벗어나서, "
                           "모델 추정이 부정확할 수 있어요 (본 적 없는 값이라 추측에 가까움).")
        else:
            w1.metric("헛스윙률 (스윙했을 때)", f"{wh0:.0%}")
            w2.metric("스윙률", f"{sw0:.0%}")
            w3.metric("진입 각도", f"{vaa_a:.1f}°")
            st.caption("🔧 사이드바의 '릴리스 포인트 바꿔 보기'로 타점·익스텐션을 바꾸면, 진입 각도가 바뀌고 "
                       "헛스윙률이 어떻게 달라지는지 모델이 계산해 줘요.")
        st.caption("2024 데이터: 높은 포심의 헛스윙률은 같은 높이 대비 진입각이 가장 가파른 공 17% → 가장 평평한 공 28%. "
                   "릴리스·진입각은 헛스윙 예측을 개선했지만(AUC 0.790→0.793), 실점 예측은 개선하지 못해서 "
                   "볼배합 순위(실점 모델)에는 넣지 않고 스윙·헛스윙 확률에만 반영했어요. "
                   "릴리스를 바꿨을 때 결과가 좋아지는지 나빠지는지(방향)는 믿을 만하지만, 모델이 실제 데이터보다 "
                   "변화 폭을 2~3배 크게 잡는 경향이 있어서 **숫자 크기는 과장될 수 있어요**.")

    # ---- 터널링
    if tr_b is not None:
        gd, gp, ydec = T.tunnel(tr_a, tr_b)
        dt_ms = (tr_b[0][-1] - tr_a[0][-1]) * 1000
        rel_gap = np.hypot(pa["release_pos_x"] - pb["release_pos_x"], pa["release_pos_z"] - pb["release_pos_z"]) * 12
        st.markdown(f"##### 🌀 터널링: {name_ko(t_a)} vs {name_ko(t_b)}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("손에서 놓을 때 간격", f"{rel_gap:.1f}인치", help="릴리스 포인트 차이. 크면 손에서부터 구종이 티 나요")
        c2.metric("타자가 결정할 때 간격", f"{gd:.1f}인치", help="도착 0.175초 전 (홈에서 약 24피트)")
        c2.caption(T.tunnel_words(gd))
        c3.metric("홈에 도착했을 때 간격", f"{gp:.1f}인치")
        c4.metric("도착 시간 차이", f"{dt_ms:+.0f}ms", help="+면 B가 더 늦게 도착. 타자 스윙 타이밍 허용 범위는 대략 ±10ms")
        msg = (f"타자가 결정할 때 두 공은 **{gd:.1f}인치** 차이로 **{T.tunnel_words(gd)}**, 홈에서는 **{gp:.1f}인치** 벌어져요. ")
        if abs(dt_ms) >= 25:
            msg += (f"게다가 {name_ko(t_b)}이(가) **{abs(dt_ms):.0f}ms {'늦게' if dt_ms > 0 else '빨리'}** 도착해서, "
                    f"같은 공인 줄 알고 휘두르면 타이밍이 어긋나요"
                    + (" → 체인지업류의 진짜 무기는 위치보다 **속도 차이**예요." if t_b in ("CH", "FS", "FO") else "."))
        if rel_gap >= 6:
            msg += f" 다만 릴리스 포인트가 {rel_gap:.1f}인치 달라서 손에서부터 티가 날 수 있어요."
        (st.success if gd < 6 else st.info)("🌀 " + msg)
        if b_mode.startswith("A와"):
            st.caption("B를 'A와 같은 터널'로 던졌다고 가정하면 결정 시점 간격은 0에 가까워요. 실제 투수는 이걸 목표로 "
                       "같은 길에서 다른 곳에 떨어뜨리려고 해요.")

    # ---- 3D 애니메이션: 투구 동작 + 공 비행 (animation.py)
    arm_deg = ph.loc[t_a, "arm_angle"] if "arm_angle" in ph.columns else np.nan
    arm_est = pd.isna(arm_deg)
    if arm_est:                                        # 팔 각도가 없으면 릴리스 높이·옆 위치로 대략 추정
        arm_deg = float(np.degrees(np.arctan2(pa["release_pos_z"] - 4.6, abs(pa["release_pos_x"]))))
    lib = load_motion_lib()
    with st.sidebar:
        st.subheader("투구 동작")
        use_real = st.radio("동작 데이터", ["실제 모션캡처 (OpenBiomechanics)", "단순 동작"], index=0 if lib else 1,
                            disabled=lib is None, help="실제: 팔 각도가 가장 비슷한 실제 투수의 모션캡처 동작을 이 투수 체격·릴리스에 맞춤"
                            ).startswith("실제")
    motion, mrow = None, None
    if use_real and lib is not None:
        mi = AN.pick_motion(lib, float(arm_deg), height_ft)
        mrow = lib["info"].loc[mi]
        motion = AN.fit_motion(lib, mi, abs(pa["release_pos_x"]), 60.5 - pa["release_pos_y"], pa["release_pos_z"], height_ft)
    st.markdown(f"##### 🎬 투구 동작 + 공 비행 (팔 각도 {arm_deg:.0f}°{' · 추정' if arm_est else ''})")
    fig = AN.make_animation(tr_a, tr_b, pa, float(arm_deg), stand, bot_z, top_z,
                            COLOR.get(t_a, "#d22d49"), COLOR.get(t_b, "#1a73e8") if tr_b is not None else "#1a73e8",
                            f"A {name_ko(t_a)}", f"B {name_ko(t_b)}" if tr_b is not None else "B",
                            height=height_ft, motion=motion)
    st.plotly_chart(fig, width="stretch")
    lg_arm = PH[(PH["pitch_type"] == t_a) & (PH["p_throws"] == hand)]["arm_angle"].mean() if "arm_angle" in PH else np.nan
    slot = ("오버핸드" if arm_deg >= 50 else "하이 스리쿼터" if arm_deg >= 38 else
            "스리쿼터" if arm_deg >= 22 else "로우 스리쿼터" if arm_deg >= 12 else "사이드암")
    st.caption("▶ 재생: 다리 들기 → 스트라이드 → 팔 젖히기 → 릴리스 → 팔로스루, 공 비행은 슬로모션 · 아래 버튼으로 카메라 바꾸기 · "
               "마우스로 돌리고 확대 가능 · 작은 마름모 = 타자가 스윙을 결정해야 하는 시점(도착 0.175초 전)")
    head = (f"🤚 {pname}의 {name_ko(t_a)} 팔 각도 {arm_deg:.0f}° → **{slot}**"
            + (f" (같은 손 리그 평균 {lg_arm:.0f}°)" if not pd.isna(lg_arm) else "") + ". ")
    if mrow is not None:
        flip = (mrow["throws"] == "L") != (hand == "L")
        st.info(head + f"투구 동작은 팔 각도가 가장 비슷한 **실제 투수의 모션캡처 동작**(OpenBiomechanics 투수 #{mrow.name + 1}: "
                f"팔 각도 {mrow['arm_angle']:.0f}°, 키 {mrow['height_ft'] * 30.48:.0f}cm, 패스트볼 {mrow['velo']:.0f}mph"
                + (", 좌우 반전" if flip else "") + f")을 {pname}의 키와 릴리스 위치에 맞춘 거예요. "
                "**동작 자체는 실제 사람의 움직임**이지만 이 투수 본인의 폼은 아니에요 (다리 드는 높이·템포 등은 다를 수 있음).")
        st.caption("모션 데이터: Driveline Baseball OpenBiomechanics Project (openbiomechanics.org) · CC BY-NC-SA 4.0, 비상업적 용도. "
                   "프로스포츠 구단·금융분석 회사의 직원/계약자는 사용할 수 없어요 (OBP_LICENSE_NOTE.txt 참고).")
    else:
        st.info(head + "투구 동작은 **릴리스 순간의 손 위치와 팔 각도만 이 투수의 실제 값**이고, 그 앞뒤 동작은 단순화한 동작이에요.")

    # ---- 2D: 옆에서 본 모습 / 타자가 보는 모습
    c1, c2 = st.columns(2)
    side = go.Figure()
    catcher = go.Figure()
    for label, color, tr, t in trs:
        side.add_trace(go.Scatter(x=tr[2], y=tr[3], mode="lines", name=label, line=dict(color=color, width=4)))
        catcher.add_trace(go.Scatter(x=tr[1], y=tr[3], mode="lines", name=label, line=dict(color=color, width=4)))
        xd, yd, zd = T.at_time(tr, tr[0][-1] - T.DECISION_T)
        for f_, xv in ((side, yd), (catcher, xd)):
            f_.add_trace(go.Scatter(x=[xv], y=[zd], mode="markers", showlegend=False,
                                    marker=dict(size=10, color="white", line=dict(color=color, width=3))))
    side.add_vrect(x0=0, x1=T.PLATE_Y, fillcolor="#9aa0a6", opacity=0.3, line_width=0)
    side.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), title="옆에서 본 모습 (왼쪽 = 홈플레이트)",
                       xaxis=dict(title="홈에서 거리 (피트)", range=[-1, 56]), yaxis=dict(title="높이 (피트)", range=[0, 7]),
                       showlegend=False)
    catcher.add_shape(type="rect", x0=-0.83, x1=0.83, y0=bot_z, y1=top_z, line=dict(color="#3c4043", width=3))
    catcher.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                          title="포수 뒤에서 본 모습 (타자 눈에 보이는 궤적)",
                          xaxis=dict(title="", range=[-3, 3], showticklabels=False, scaleanchor="y"),
                          yaxis=dict(title="높이 (피트)", range=[0, 7]), showlegend=False)
    c1.plotly_chart(side, width="stretch")
    c2.plotly_chart(catcher, width="stretch")
    a = ars_all[(ars_all["pitcher"] == pid) & (ars_all["pitch_type"] == t_a)]
    if len(a):
        a = a.iloc[0]
        st.caption(f"{name_ko(t_a)}: 회전수 {a['release_spin_rate']:.0f}rpm · 좌우 무브먼트 {a['pfx_x']:+.1f}인치 · "
                   f"수직 무브먼트 {a['pfx_z']:+.1f}인치 (중력만 받은 공 대비, 투수 팔 방향 기준)")
    st.caption("궤적은 이 투수 이 구종의 2024년 평균 릴리스 지점·속도·가속도(회전·공기저항 효과)를 그대로 쓰고, "
               "목표 코스에 도착하도록 던지는 방향만 조절해서 계산했어요.")
