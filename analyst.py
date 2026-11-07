"""
🤖 AI 분석관 — 언어 모델(Claude)에 지금까지 만든 모델들을 '도구'로 쥐여주고 자연어로 묻고 답하기

구조
  사용자 질문 → Claude가 필요한 도구를 고름 (선수 찾기, 매치업 시뮬레이션, 공 모양 목표, 스윙 목표 ...)
            → 이 파일의 Analyst가 실제 모델을 돌려 숫자를 돌려줌
            → Claude는 그 숫자만 가지고 해석·설명 (숫자를 지어내지 않게 시스템 프롬프트로 강제)

API 키: Anthropic 콘솔에서 발급 → .streamlit/secrets.toml 에  ANTHROPIC_API_KEY = "sk-ant-..."
        (또는 환경변수 ANTHROPIC_API_KEY). 키는 절대 깃허브 등에 올리지 않기.
"""
import json
import unicodedata

import numpy as np
import pandas as pd

import batter_mech as BM
import mechanics as MC
import recommend as R

DEFAULT_MODEL = "claude-sonnet-5-5"
PITCH_KO = {"FF": "포심", "SI": "싱커", "FC": "커터", "SL": "슬라이더", "ST": "스위퍼", "CU": "커브", "KC": "너클 커브",
            "SV": "슬러브", "CS": "슬로 커브", "CH": "체인지업", "FS": "스플리터", "FO": "포크볼", "SC": "스크루볼",
            "KN": "너클볼", "EP": "이퓨스"}
X_C = [-1.1, -0.55, 0.0, 0.55, 1.1]          # 코스 칸 중심 (타자 기준, 피트)
Z_C = [-0.2, 0.15, 0.5, 0.85, 1.2]           # 높이 칸 중심 (존 바닥 0 ~ 꼭대기 1)

SYSTEM = """당신은 KBO·MLB 구단의 데이터 분석관입니다. 사용자는 한국어로 야구 질문을 합니다.

규칙
1. 선수 성적·추천·확률 같은 숫자는 반드시 도구를 호출해서 얻은 값만 쓰세요. 기억이나 추측으로 숫자를 만들지 마세요.
2. 선수 이름은 영어 성(last name)으로 find_player를 먼저 호출해 ID를 확인하세요 (예: 저지 → "Judge", 오타니 → "Ohtani").
   후보가 여럿이면 가장 그럴듯한 선수를 고르고 누구로 판단했는지 밝히세요. 데이터에 없으면 없다고 말하세요.
3. 데이터는 MLB 스탯캐스트(투구·매치업 2024, 타자 스윙 2025)입니다. KBO 선수는 데이터가 없습니다.
4. 검증 결과를 정직하게 반영하세요 (validation_notes 도구 참고):
   - 코스 추천은 2025 데이터로 검증됨. 구종 순위는 검증되지 않음(타자가 노리면 달라지는 게임이론 문제) → 구종 질문엔 pitch_mix 도구로 '섞어 던지기 비율'까지 같이 답하세요.
   - 공 모양(무브·구속) 변화 효과는 검증됨(상관 0.29). 릴리스 포인트 변화 효과는 약함(0.09).
   - 타자 스윙 변화 효과는 검증됨(헛스윙 0.42, 타구 질 0.48).
5. 답변은 한국어로, 결론 먼저, 짧고 구체적으로. 숫자에는 단위와 의미를 붙이세요
   (rv100 = 100구 던졌을 때 상대 팀 득점 변화, 음수일수록 투수에게 좋음 / xwOBA 포인트 = 0.001).
6. 코스는 타자 기준(몸쪽/바깥쪽)으로 말하고, 필요하면 사용자의 질문 의도(수비 팀/공격 팀 입장)에 맞춰 설명하세요."""

TOOLS = [
    {"name": "find_player", "description": "선수 이름(영어 성 또는 일부)으로 ID 찾기. 다른 도구를 쓰기 전에 먼저 호출.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "영어 이름 일부, 예: 'Judge', 'Skenes'"},
         "role": {"type": "string", "enum": ["pitcher", "batter", "any"]}}, "required": ["name"]}},
    {"name": "pitcher_report", "description": "투수의 구종별 구사율·구속·무브먼트·헛스윙·실점 가치·리그 백분위와 제일 좋은 공",
     "input_schema": {"type": "object", "properties": {"pitcher_id": {"type": "integer"}}, "required": ["pitcher_id"]}},
    {"name": "batter_report", "description": "타자의 핫존·콜드존, 구종 계열별 약점, 유인구 스윙률, 타구 성향, 스윙(배트 트래킹) 지표",
     "input_schema": {"type": "object", "properties": {"batter_id": {"type": "integer"}}, "required": ["batter_id"]}},
    {"name": "pitch_recommendation",
     "description": "투수 vs 타자, 카운트·아웃·주자 상황에서 어떤 구종을 어느 코스로 던지면 실점 기대가 가장 낮은지 (수비 팀 관점)",
     "input_schema": {"type": "object", "properties": {
         "pitcher_id": {"type": "integer"}, "batter_id": {"type": "integer"},
         "balls": {"type": "integer", "minimum": 0, "maximum": 3}, "strikes": {"type": "integer", "minimum": 0, "maximum": 2},
         "outs": {"type": "integer", "minimum": 0, "maximum": 2},
         "runners": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3]}, "description": "주자 있는 루, 예: [1,3]"},
         "prev_pitch_type": {"type": "string", "description": "직전 공 구종 코드 (예: FF). 모르면 생략"},
         "prev_location": {"type": "string", "description": "직전 공 코스, 예: '높게 몸쪽'"}},
         "required": ["pitcher_id", "batter_id", "balls", "strikes"]}},
    {"name": "hitting_plan",
     "description": "타자 입장: 이 투수 상대로 이 카운트에서 노려야 할 공(치면 이득)과 참아야 할 공(치면 손해) (공격 팀 관점)",
     "input_schema": {"type": "object", "properties": {
         "batter_id": {"type": "integer"}, "pitcher_id": {"type": "integer"},
         "balls": {"type": "integer"}, "strikes": {"type": "integer"}}, "required": ["batter_id", "pitcher_id", "balls", "strikes"]}},
    {"name": "pitch_shape_targets",
     "description": "투수 개선: 구종별로 공 모양(수직·가로 무브, 구속)을 어떻게 바꾸면 헛스윙이 느는지, 릴리스 조정으로 가능한 만큼",
     "input_schema": {"type": "object", "properties": {"pitcher_id": {"type": "integer"}}, "required": ["pitcher_id"]}},
    {"name": "batter_swing_targets",
     "description": "타자 개선: 배트 스피드·어택 앵글·컨택 포인트·스윙 방향을 어떻게 바꾸면 좋은지 (타자 유형별 기준)",
     "input_schema": {"type": "object", "properties": {
         "batter_id": {"type": "integer"},
         "focus": {"type": "string", "enum": ["style", "xw_sw", "xwcon", "miss"],
                   "description": "style=타자 유형에 맞게(기본), xw_sw=종합, xwcon=타구 질, miss=헛스윙 줄이기"}},
         "required": ["batter_id"]}},
    {"name": "pitch_mix",
     "description": "게임이론 볼배합(혼합 전략): 타자가 노리고 들어오는 효과를 넣어 이 상황의 구종별 추천 비율(%)과 평소 비율 비교. "
                    "'뭘 던질까'의 구종 질문엔 pitch_recommendation(코스)과 함께 이걸 쓰세요",
     "input_schema": {"type": "object", "properties": {
         "pitcher_id": {"type": "integer"}, "batter_id": {"type": "integer"},
         "balls": {"type": "integer"}, "strikes": {"type": "integer"}, "outs": {"type": "integer"},
         "runners": {"type": "array", "items": {"type": "integer"}},
         "beta": {"type": "number", "description": "타자가 노리는 강도. 생략하면 MLB 투수 행동에 맞춘 기본값"}},
         "required": ["pitcher_id", "batter_id", "balls", "strikes"]}},
    {"name": "minimax_check",
     "description": "MLB 투수들이 미니맥스(게임이론 균형)대로 던지는지 검증 결과 — 리그 전체, 카운트별, (투수를 주면) 그 투수의 패스트볼 편중 여부",
     "input_schema": {"type": "object", "properties": {"pitcher_id": {"type": "integer"}}}},
    {"name": "similar_players",
     "description": "신경망 임베딩(공에 반응하는 방식)으로 이 선수와 가장 비슷한 선수들 찾기. 스카우팅·비교·신인 예측에 활용",
     "input_schema": {"type": "object", "properties": {
         "player_id": {"type": "integer"}, "role": {"type": "string", "enum": ["pitcher", "batter"]},
         "k": {"type": "integer", "minimum": 3, "maximum": 15}}, "required": ["player_id", "role"]}},
    {"name": "validation_notes", "description": "이 시스템의 각 기능이 실제 데이터로 얼마나 검증됐는지 요약",
     "input_schema": {"type": "object", "properties": {}}},
]


def _plain(t):
    t = unicodedata.normalize("NFKD", str(t))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


class Analyst:
    """도구 실행기 — 앱이 이미 불러온 모델들을 받아서 씀"""

    def __init__(self, M, S, O=None, P=None, report=None, samples=None, swing_v2=None, move=None, BS=None, EMB=None):
        self.M, self.S, self.O, self.P = M, S, O, P
        self.rep, self.samp, self.sm, self.move, self.BS, self.EMB = report, samples, swing_v2 or S, move, BS, EMB
        ars = M["arsenal"]
        self.pitchers = ars.drop_duplicates("pitcher").set_index("pitcher")[["player_name", "p_throws"]]
        names = dict(M["batter_names"])
        self.batters = names

    # ---------------------------------------------------------------- 도구들
    def find_player(self, name, role="any"):
        q = _plain(name)
        out = []
        if role in ("pitcher", "any"):
            for pid, r in self.pitchers.iterrows():
                if q in _plain(r["player_name"]):
                    out.append({"id": int(pid), "name": r["player_name"], "role": "투수",
                                "hand": "우투" if r["p_throws"] == "R" else "좌투"})
        if role in ("batter", "any"):
            stand = self.M["batter_stand"]
            for bid, nm in self.batters.items():
                if q in _plain(nm) and bid in self.M["batter_zone"].index:
                    s = stand.get(bid, "R")
                    out.append({"id": int(bid), "name": nm, "role": "타자", "hand": {"R": "우타", "L": "좌타"}.get(s, s)})
        out.sort(key=lambda d: (not _plain(d["name"]).startswith(q), d["name"]))
        return {"candidates": out[:10], "count": len(out)}

    def pitcher_report(self, pitcher_id):
        if self.rep is None:
            return {"error": "pitcher_report.pkl 없음"}
        r = self.rep[self.rep["pitcher"] == pitcher_id].sort_values("usage", ascending=False)
        if r.empty:
            return {"error": "이 투수의 리포트가 없어요 (2024 투구 수 부족)"}
        rows = []
        for _, x in r[r["usage"] >= 0.03].iterrows():
            rows.append({"구종": f"{x['pitch_type']} ({PITCH_KO.get(x['pitch_type'], x['pitch_type'])})",
                         "구사율%": _r(x["usage"] * 100, 0), "구속mph": _r(x["velo"], 1),
                         "수직무브cm": _r(x["ivb"] * 2.54, 0), "가로무브cm(팔쪽+)": _r(-x["hb"] * 2.54, 0),
                         "회전rpm": _r(x["spin"], 0), "헛스윙%(투구당)": _r(x["swstr"] * 100, 1),
                         "100구당실점": _r(x["rv100"], 2), "구속백분위": _r(x["velo_pct"], 0),
                         "헛스윙백분위": _r(x["swstr_pct"], 0), "실점억제백분위": _r(x["rv100_pct"], 0),
                         "제일좋은공": bool(x["is_best"])})
        return {"name": r["player_name"].iloc[0], "hand": r["p_throws"].iloc[0], "season": 2024,
                "pitches": rows, "note": "백분위는 같은 구종 100구 이상 MLB 투수 중 위치(높을수록 좋음)"}

    def batter_report(self, batter_id):
        M = self.M
        if batter_id not in M["batter_zone"].index:
            return {"error": "이 타자 데이터가 없어요"}
        p = M["profile"]
        c = p["cell"][p["cell"]["batter"] == batter_id].set_index("cell")["b_cell_rv"]
        lg = p["cell_league"]
        zone = []
        for cell, v in c.items():
            xi, zi = divmod(int(cell), 5)
            if 1 <= xi <= 3 and 1 <= zi <= 3:              # 스트라이크존 안 9칸
                zone.append({"코스": R.describe(X_C[xi], Z_C[zi]), "리그대비": (v - lg.get(cell, 0)) * 100})
        zone.sort(key=lambda d: d["리그대비"], reverse=True)
        oz = p["oz"][p["oz"]["batter"] == batter_id].set_index("oz")["b_swing_oz"]
        ozn = {0: "존 안", 1: "낮게 빠지는 공", 2: "높게 빠지는 공", 3: "바깥쪽 빠지는 공", 4: "몸쪽 빠지는 공"}
        grp = p["group"][p["group"]["batter"] == batter_id].set_index("group")
        ov = p["overall"].set_index("batter").loc[batter_id] if batter_id in set(p["overall"]["batter"]) else None
        out = {"name": self.batters.get(batter_id), "stand": M["batter_stand"].get(batter_id),
               "핫존(리그 대비 100구당 득점 +)": [{"코스": z["코스"], "리그대비": _r(z["리그대비"], 2)} for z in zone[:3]],
               "콜드존": [{"코스": z["코스"], "리그대비": _r(z["리그대비"], 2)} for z in zone[-3:]],
               "구역별 스윙률%": {ozn[k]: _r(v * 100, 0) for k, v in oz.items()},
               "구종 계열별": {{"fastball": "패스트볼 계열", "breaking": "변화구", "offspeed": "체인지업·스플리터"}.get(g, g): {"헛스윙%": _r(r_["b_whiff"] * 100, 0), "유인구스윙%": _r(r_["b_chase"] * 100, 0),
                              "xwOBA": _r(r_["b_xwoba"], 3), "땅볼%": _r(r_["b_gb_g"] * 100, 0)} for g, r_ in grp.iterrows()},
               "season": "2024 (축소 보정된 값)"}
        if ov is not None:
            out["타구"] = {"평균타구속도mph": _r(ov.get("b_ev"), 1), "평균발사각": _r(ov.get("b_la"), 1),
                         "당긴타구%": _r(ov.get("b_pull", np.nan) * 100, 0), "밀어친타구%": _r(ov.get("b_oppo", np.nan) * 100, 0)}
        if self.BS is not None and batter_id in self.BS["report"].index:
            r = self.BS["report"].loc[batter_id]
            out["스윙(2025 배트트래킹)"] = {
                "유형": BM.archetype(r), "배트스피드mph": _r(r["bat_speed"], 1), "배트스피드백분위": _r(r["bat_speed_pct"], 0),
                "어택앵글": _r(r["attack_angle"], 1), "스윙방향(-당기기)": _r(r["attack_direction"], 1),
                "컨택포인트cm(몸 앞)": _r(r["contact_y"] * 2.54, 0), "헛스윙%(스윙당)": _r(r["whiff"] * 100, 1),
                "타구질(인플레이 xwOBA)": _r(r["xwcon"], 3)}
        return out

    def _stand(self, pitcher_id, batter_id):
        s = self.M["batter_stand"].get(batter_id, "R")
        if s not in ("R", "L"):                          # 스위치 히터: 투수 반대 손
            s = "L" if self.pitchers.loc[pitcher_id, "p_throws"] == "R" else "R"
        return s

    def pitch_recommendation(self, pitcher_id, batter_id, balls, strikes, outs=0, runners=None, prev_pitch_type=None,
                             prev_location=None):
        if pitcher_id not in self.pitchers.index or batter_id not in self.M["batter_zone"].index:
            return {"error": "투수나 타자 데이터가 없어요"}
        stand = self._stand(pitcher_id, batter_id)
        runners = runners or []
        sit = (outs, int(1 in runners), int(2 in runners), int(3 in runners))
        prev = None
        if prev_pitch_type:
            a = self.M["arsenal"]
            a = a[(a["pitcher"] == pitcher_id) & (a["pitch_type"] == prev_pitch_type)]
            if len(a):
                prev = a.iloc[0].to_dict()
                prev["x"], prev["z"] = R.parse_spot(prev_location)
        res = R.simulate(self.M, pitcher_id, batter_id, balls, strikes, stand, 0.5, prev, self.S, sit, n_noise=25)
        res = res[res["usage"] >= 0.04]
        fmt = lambda x: {"구종": f"{x['pitch_type']} ({PITCH_KO.get(x['pitch_type'], '')})", "코스": R.describe(x["tx"], x["tz"]),
                         "100구당실점": _r(x["rv100"], 2), "스윙확률%": _r(x.get("p_swing", np.nan) * 100, 0),
                         "헛스윙확률%(스윙시)": _r(x.get("p_whiff", np.nan) * 100, 0)}
        res = res.assign(spot=[R.describe(x, z) for x, z in zip(res["tx"], res["tz"])])
        best = res.sort_values("rv100").drop_duplicates(["pitch_type", "spot"])   # 같은 말로 표현되는 칸은 하나만
        per = best.drop_duplicates("pitch_type")
        return {"상황": f"{balls}-{strikes}, {outs}아웃, 주자 {runners or '없음'}, 타석 {'우타' if stand == 'R' else '좌타'}",
                "추천 Top5 (구종+코스)": [fmt(x) for _, x in best.head(5).iterrows()],
                "구종별 최선 코스": [dict(fmt(x), 구사율=_r(x["usage"] * 100, 0)) for _, x in per.iterrows()],
                "피해야 할 곳 Top3": [fmt(x) for _, x in best.tail(3).iloc[::-1].iterrows()],
                "note": "100구당실점: 음수일수록 투수에게 좋음. 제구 오차 ±15cm 가정. 코스는 검증됨, 구종 순위는 미검증(섞어 던지기 고려)."}

    def hitting_plan(self, batter_id, pitcher_id, balls, strikes):
        if self.O is None:
            return {"error": "offense_model.pkl 없음"}
        stand = self._stand(pitcher_id, batter_id)
        out = []
        for key in ("swing_rv", "take_rv"):
            m = dict(self.M, model=self.O[key], features=self.O["features"])
            r = R.simulate(m, pitcher_id, batter_id, balls, strikes, stand, 0.15, None, None, (0, 0, 0, 0), n_noise=12,
                           extra={"if_al": 0, "of_al": 0})
            out.append(r.set_index(["pitch_type", "tx", "tz"])["rv100"])
        res = out[0].rename("swing").to_frame().join(out[1].rename("take")).reset_index()
        res["value"] = res["swing"] - res["take"]
        u = self.M["arsenal"][self.M["arsenal"]["pitcher"] == pitcher_id].set_index("pitch_type")["usage"]
        res["usage"] = res["pitch_type"].map(u)
        res = res[res["usage"] >= 0.04]
        fmt = lambda x: {"구종": f"{x['pitch_type']} ({PITCH_KO.get(x['pitch_type'], '')})", "코스": R.describe(x["tx"], x["tz"]),
                         "치면 이득(100구당 득점)": _r(x["value"], 2), "투수 구사율%": _r(x["usage"] * 100, 0)}
        res = res.assign(spot=[R.describe(x, z) for x, z in zip(res["tx"], res["tz"])])
        hunt = (res.assign(w=res["value"] * res["usage"]).sort_values("w", ascending=False)
                   .drop_duplicates(["pitch_type", "spot"]))
        lay = res.sort_values("value").drop_duplicates(["pitch_type", "spot"])
        return {"상황": f"{balls}-{strikes}",
                "노릴 공 Top4 (이득×구사율)": [fmt(x) for _, x in hunt.head(4).iterrows()],
                "참을 공 Top4 (치면 손해)": [fmt(x) for _, x in lay.head(4).iterrows()],
                "note": "치면 이득 = (쳤을 때 득점 기대) − (그냥 봤을 때 득점 기대), 100구당. +면 치는 게 이득."}

    def pitch_shape_targets(self, pitcher_id):
        if self.rep is None or self.samp is None or self.P is None:
            return {"error": "투수 개선 데이터 없음"}
        hand = self.pitchers.loc[pitcher_id, "p_throws"] if pitcher_id in self.pitchers.index else "R"
        ph = self.P[self.P["pitcher"] == pitcher_id].set_index("pitch_type")
        if not (self.samp["pitcher"] == pitcher_id).any():
            return {"error": "이 투수는 샘플이 없어요 (2024 300구 이상만)"}
        sres, types = MC.shape_simulate(self.sm, self.samp, ph, self.rep, pitcher_id, hand)
        rows = []
        for t in types:
            b = MC.best_shape(sres, t)
            b = b.iloc[0] if len(b) else None
            if b is None or b["score"] <= 0:
                rows.append({"구종": t, "추천": "지금 그대로"})
                continue
            rows.append({"구종": f"{t} ({PITCH_KO.get(t, '')})", "수직무브cm": _r(b["d_ivb"] * 2.54, 0),
                         "가로무브cm(팔쪽+)": _r(b["d_arm"] * 2.54, 0), "구속mph": _r(b["d_velo"], 0),
                         "그 구종 헛스윙%p": _r(b["d"], 2), "다른 구종 영향%p": _r(b.get("d_others", 0), 2)})
        out = {"공 모양 목표": rows, "note": "공 모양 변화 → 헛스윙 효과는 2025 실제 변화로 검증(상관 0.29)."}
        if self.move is not None:
            res, types_, usage = MC.simulate(self.sm, self.samp, ph, self.rep, pitcher_id, hand, move=self.move)
            bp = MC.best_plan(res, "all", types_)
            if len(bp):
                x = bp.iloc[0]
                out["릴리스로 가능한 만큼(전체 구종)"] = {
                    "변화": MC.move_words(x["dh"], x["ds"], x["de"]) if hasattr(MC, "move_words") else
                    f"타점 {x['dh'] * 30.48:+.0f}cm, 옆 {x['ds'] * 30.48:+.0f}cm, 익스텐션 {x['de'] * 30.48:+.0f}cm",
                    "전체 헛스윙%p": _r(x["d_all"], 2), "note": "릴리스→헛스윙 경로는 검증 상관 0.09로 약함 — 참고용"}
        return out

    def batter_swing_targets(self, batter_id, focus="style"):
        if self.BS is None or batter_id not in self.BS["report"].index:
            return {"error": "이 타자의 배트 트래킹 데이터가 없어요 (2025 200스윙 이상만)"}
        r = self.BS["report"].loc[batter_id]
        arch = BM.archetype(r)
        res = BM.simulate(self.BS, batter_id)
        out = {"유형": arch, "기준": BM.STYLE_RULE[arch]["desc"] if focus == "style" else BM.FOCUS.get(focus)}
        for lab, only in (("추천", None), ("배트스피드 그대로", ["d_aa", "d_cy", "d_dir"])):
            b = BM.best(res, focus, only=only, arch=arch)
            if b.empty:
                continue
            b = b.iloc[0]
            if not b["meaningful"] or (b[["d_bs", "d_aa", "d_cy", "d_dir"]] == 0).all():
                out[lab] = {"판단": "지금 스타일 유지 (바꿔도 차이 미미)", "참고": BM.words(b, self.BS["k_len"])}
            else:
                out[lab] = {"변화": BM.words(b, self.BS["k_len"]), "헛스윙%p": _r(b["d_miss"], 1),
                            "타구질(xwOBA포인트)": _r(b["d_xwcon"], 0), "스윙당(xwOBA포인트)": _r(b["d_xw_sw"], 1)}
        out["note"] = "스윙 변화→결과 효과는 2024→2025 타자 286명으로 검증(헛스윙 0.42, 타구 질 0.48). 같은 시즌 동시 변화라 인과는 아님."
        return out

    def pitch_mix(self, pitcher_id, batter_id, balls, strikes, outs=0, runners=None, beta=None):
        import game as GM
        if pitcher_id not in self.pitchers.index or batter_id not in self.M["batter_zone"].index:
            return {"error": "투수나 타자 데이터가 없어요"}
        g, _ = GM.load()
        beta = float(beta) if beta is not None else (float(g["beta"]) if g else GM.DEFAULT_BETA)
        runners = runners or []
        sit = (outs, int(1 in runners), int(2 in runners), int(3 in runners))
        res = R.simulate(self.M, pitcher_id, batter_id, balls, strikes, self._stand(pitcher_id, batter_id), 0.5, None,
                         None, sit, n_noise=20)
        df, ev = GM.plan(res, beta)
        return {"상황": f"{balls}-{strikes}, {outs}아웃, 주자 {runners or '없음'}", "beta": beta,
                "구종별": [{"구종": f"{r['pitch_type']} ({PITCH_KO.get(r['pitch_type'], '')})", "평소%": _r(r["usage"] * 100, 0),
                          "추천%": _r(r["mix"] * 100, 0), "이 공 기대실점(100구당)": _r(r["rv0"], 2),
                          "노릴 코스": R.describe(r["tx"], r["tz"])} for r in df.sort_values("mix", ascending=False).to_dict("records")],
                "기대실점(100구당, 이 가정 아래)": {k: _r(v, 2) for k, v in ev.items()},
                "note": "β = 평소보다 많이 던질 때 노림당하는 정도. 기본값은 MLB 투수들의 실제 카운트별 선택에 가장 잘 맞는 값(역 게임이론). "
                        "타자가 예측 가능한 공에 헛스윙이 줄어드는 건 2024·2025 모두 확인됨. 추천 비율 자체가 실점을 줄이는지는 미검증."}

    def minimax_check(self, pitcher_id=None):
        import game as GM
        mm = GM.load_minimax()
        if mm is None:
            return {"error": "minimax.pkl 없음"}
        L = mm["league"]
        out = {"리그(같은 투수·카운트·좌우 안, 100구당 실점 차)": {
            str(yr): {"변화구−패스트볼": [_r(L[yr]["g1"][0]), [_r(x) for x in L[yr]["g1"][1]]],
                      "오프스피드−패스트볼": [_r(L[yr]["g2"][0]), [_r(x) for x in L[yr]["g2"][1]]]} for yr in (2024, 2025)},
            "두 시즌 모두 확실한 차이": GM.replicated(mm).round(2).to_dict("records"),
            "투수별 차이의 시즌 간 상관": _r(mm["persist"]["r_raw"], 3),
            "note": "음수 = 그 계열이 패스트볼보다 실점 적음. 전체로는 차이가 작아 미니맥스에 가까움. 0-2 오프스피드만 두 시즌 모두 패스트볼보다 확실히 좋음. "
                    "투수 개인 차이는 거의 운(시즌 간 상관 0.11)."}
        if pitcher_id is not None:
            pg = GM.pitcher_gap(mm, pitcher_id)
            out["이 투수 패스트볼−나머지(축소 보정)"] = None if pg is None else {k: _r(v, 2) for k, v in pg.items()}
        return out

    def similar_players(self, player_id, role, k=8):
        if self.EMB is None:
            return {"error": "embed_model.npz 없음"}
        import embed as EM
        sim = EM.similar(self.EMB, role, player_id, k=k, min_n=500)
        if sim is None:
            return {"error": "이 선수의 임베딩이 없어요"}
        name = (lambda i, r: self.batters.get(int(i), f"ID {int(i)}")) if role == "batter" else                (lambda i, r: r.get("name") if isinstance(r.get("name"), str) else f"ID {int(i)}")
        cols = ["whiff", "chase", "ev", "rv100"] if role == "batter" else ["velo", "whiff", "fb_share", "rv100"]
        rows = [{"선수": name(r["id"], r), "유사도": _r(r["sim"], 2),
                 **{EM.STAT_KO[c]: _r(r[c], 3) for c in cols if c in r}} for r in sim.to_dict("records")]
        note = ("신경망이 2024~2025 투구로 배운 16차원 성향 벡터의 코사인 유사도 (500구 이상 선수 중). "
                "헛스윙 예측 정확도는 기존 모델보다 높음(AUC 0.792 vs 0.787, 2025 검증).")
        if role == "pitcher":
            note += (" 주의: 투수 임베딩은 구속·무브 같은 공 정보를 뺀 나머지(디셉션·터널 등)를 담아서 "
                     "'구종 구성이 비슷한 투수'와는 다름.")
        return {"similar": rows, "note": note}

    def validation_notes(self):
        return {"코스 추천": "2025 실제 투구로 백테스트 — 추천 코스가 실제로 실점이 적었음 (검증됨)",
                "구종 순위": "검증 안 됨: 좋은 공만 던지면 타자가 노림(게임이론). 섞어 던지기 관점 필요",
                "높은 직구 가설": "2024에서 보였지만 2025에서 재현 안 됨",
                "릴리스 포인트 직접 효과": "−0.06 (틀림) → 폐기. 릴리스→무브 경로로만 0.09 (약함)",
                "공 모양(무브·구속) 효과": "0.29 (가능한 최대 ≈0.56), 패스트볼 대비 피처로 개선",
                "투수 본인 릴리스→무브 관계": "리그 평균과 차이 없음 → 미채택",
                "타자 스윙 효과": "헛스윙 0.42, 타구 질 0.48, 스윙당 0.26 (286명)",
                "타자 유형별": "컨택형은 어택 앵글 올려도 이득 없음, 파워형은 이득 — 모델은 파워형 이득을 작게 봄",
                "섞어 던지기(게임이론)": "타자는 예측 가능한 공에 헛스윙이 줄어듦(2024 −0.38, 2025 −0.15%p per +10%p 구사율). "
                                   "β는 MLB 투수 실제 선택으로 역추정. 비율 추천 자체는 미검증",
                "미니맥스 검증": "MLB 전체로는 구종 간 결과 차이가 100구당 0.1~0.3점으로 작아 균형에 가까움. 0-2 오프스피드만 두 시즌 모두 확실히 저평가(더 던질 여지)",
                "선수 임베딩(신경망)": "2025 헛스윙 AUC 0.792 vs 기존 0.787, 둘 평균 0.797. 표본 적은 타자는 기존과 비슷"}

    # ---------------------------------------------------------------- 실행
    def run(self, name, args):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            return {"error": f"모르는 도구: {name}"}
        try:
            return fn(**args)
        except Exception as e:                       # 모델이 잘못된 인자를 줘도 대화는 계속되게
            return {"error": f"{type(e).__name__}: {e}"}


def _block_dict(b):
    """API에 다시 보낼 수 있는 최소 형태로 (불필요한 필드 제거)"""
    d = b.model_dump() if hasattr(b, "model_dump") else dict(b)
    if d.get("type") == "text":
        return {"type": "text", "text": d.get("text", "")}
    if d.get("type") == "tool_use":
        return {"type": "tool_use", "id": d["id"], "name": d["name"], "input": d.get("input") or {}}
    return d


def chat(client, analyst, messages, model=DEFAULT_MODEL, max_rounds=10, on_tool=None):
    """messages: [{'role','content'}...] (마지막은 사용자 질문). 도구를 쓰며 답을 완성.
    돌려줌: (최종 답변 텍스트, 도구 사용 기록, 갱신된 messages)"""
    msgs = list(messages)
    log = []
    for _ in range(max_rounds):
        resp = client.messages.create(model=model, max_tokens=2500, system=SYSTEM, tools=TOOLS, messages=msgs)
        content = [_block_dict(b) for b in resp.content]
        msgs.append({"role": "assistant", "content": content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            return text, log, msgs
        results = []
        for b in content:
            if b.get("type") != "tool_use":
                continue
            out = analyst.run(b["name"], b.get("input") or {})
            log.append({"tool": b["name"], "input": b.get("input"), "output": out})
            if on_tool:
                on_tool(b["name"], b.get("input"))
            results.append({"type": "tool_result", "tool_use_id": b["id"],
                            "content": json.dumps(out, ensure_ascii=False, default=str)})
        msgs.append({"role": "user", "content": results})
    return "도구 호출이 너무 많아서 멈췄어요. 질문을 조금 더 구체적으로 해 주세요.", log, msgs
