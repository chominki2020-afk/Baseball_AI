"""
개인화(1: 투수 본인 릴리스→무브, 2: 자기 패스트볼 대비 피처) 검증 — validate_release.py와 같은 자연 실험

2024 정보만으로 예측 → 2024→2025 실제로 릴리스를 바꾼 투수들의 실제 헛스윙 변화와 비교
  리그 무브    : 리그 평균 릴리스→무브 관계 (2024 경기별)            ← 지금 앱 방식
  개인 무브    : 그 투수 본인의 릴리스→무브 관계 (2024 경기별, 리그 쪽으로 축소)
  실제 공 모양  : 2025에 실제로 바뀐 무브·구속·회전 (스윙 모델 자체 점검)
각각 기존 스윙 모델(v1)과 패스트볼 대비 피처 모델(v2)로

실행: python validate_personal.py   (필요: val_release_prep.pkl, personal_move.pkl, swing_model(_v2).pkl, pitch_samples(_v2).pkl, physics.pkl)
"""
import pickle

import numpy as np
import pandas as pd

import build_release_move as RM
import mechanics as MC
from validate_release import corr_ci, CHANGE

P = pickle.load(open("val_release_prep.pkl", "rb"))
PM = pickle.load(open("personal_move.pkl", "rb"))
phys = pickle.load(open("physics.pkl", "rb")).set_index(["pitcher", "pitch_type"])
MODELS = {"v1": (pickle.load(open("swing_model.pkl", "rb")), pickle.load(open("pitch_samples.pkl", "rb"))),
          "v2": (pickle.load(open("swing_model_v2.pkl", "rb")), pickle.load(open("pitch_samples_v2.pkl", "rb")))}

a = P["a"][P["a"]["n"] >= 100].set_index(["pitcher", "pitch_type"])
b = P["b"][P["b"]["n"] >= 100].set_index(["pitcher", "pitch_type"])
j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
for c in RM.X + RM.Y + ["swstr"]:
    j["d_" + c] = j[c + "_b"] - j[c + "_a"]
for c in ["d_swstr"] + ["d_" + x for x in RM.X]:
    j[c] = j[c] - j.groupby(level="pitch_type")[c].transform("median")
j["moved"] = j[["d_rel_z", "d_rel_side", "d_ext"]].abs().max(axis=1) >= CHANGE
keys = set(map(tuple, MODELS["v1"][1][["pitcher", "pitch_type"]].drop_duplicates().to_numpy())) & \
       set(map(tuple, MODELS["v2"][1][["pitcher", "pitch_type"]].drop_duplicates().to_numpy()))
j = j[[k in keys and k in phys.index for k in j.index]]
j["act"] = j["d_swstr"] * 100
j["se"] = 100 * np.sqrt(j["swstr_a"] * (1 - j["swstr_a"]) / j["n_a"] + j["swstr_b"] * (1 - j["swstr_b"]) / j["n_b"])
print(f"대상 {len(j)}개, 릴리스 바꾼 경우 {j['moved'].sum()}개")


def move_pred(k, kind):
    t = k[1]
    dx = np.array([j.loc[k, "d_rel_z"], j.loc[k, "d_rel_side"], j.loc[k, "d_ext"]])
    if kind == "personal" and k in PM["val2024"]:
        c = PM["val2024"][k]
        return float(c["ivb"] @ dx), float(c["hb"] @ dx)
    m = MC.move_delta(PM["prior2024"], t, *dx)
    return m["ivb"], m["hb"]


# 릴리스→무브 관계 자체: 리그 vs 개인
mv = j[j["moved"]]
for kind in ("league", "personal"):
    pr = np.array([move_pred(k, kind) for k in mv.index])
    print(f"릴리스→무브 [{kind:8s}] 수직 상관 {np.corrcoef(pr[:, 0], mv['d_ivb'])[0, 1]:+.3f}, "
          f"가로 상관 {np.corrcoef(pr[:, 1], mv['d_hb'])[0, 1]:+.3f}, 예측 크기 {np.abs(pr[:, 0]).mean():.2f}in "
          f"(실제 {mv['d_ivb'].abs().mean():.2f}in)  개인 계수 있는 비율 {np.mean([k in PM['val2024'] for k in mv.index]):.0%}")

# 투구 단위 변화량
deltas = {}
for kind in ("league", "personal", "actual"):
    dd = {}
    for k in j.index:
        if kind == "actual":
            iv, hb, v, sp = j.loc[k, "d_ivb"], j.loc[k, "d_hb"], j.loc[k, "d_velo"], j.loc[k, "d_spin"]
        else:
            iv, hb = move_pred(k, kind)
            v = sp = 0.0
        dd[k] = (iv, hb, v, sp, MC.vaa_shift(phys.loc[k].to_dict(), 0, 0, 0, d_ivb=iv))
    deltas[kind] = dd


def predict(ver, dd):
    S, samp = MODELS[ver]
    x = samp[[(p, t) in dd for p, t in zip(samp["pitcher"], samp["pitch_type"])]].copy()
    X0 = x[S["features"]]
    p0 = S["swing"].predict_proba(X0)[:, 1] * S["whiff"].predict_proba(X0)[:, 1]
    k = list(zip(x["pitcher"], x["pitch_type"]))
    arr = np.array([dd[kk] for kk in k])
    x["pfx_z"] += arr[:, 0]
    x["pfx_x"] += arr[:, 1]
    x["release_speed"] += arr[:, 2]
    x["release_spin_rate"] += arr[:, 3]
    x["vaa_rel"] += arr[:, 4]
    if "rel" in S:                                          # 주력 패스트볼이 바뀌면 기준값도 같이 이동
        fbk = list(zip(x["pitcher"], x["main_fb"]))
        sh = np.array([dd.get(kk, (0, 0, 0, 0, 0)) for kk in fbk])
        x["fb_dv"] = x["release_speed"] - (x["fb_velo_ref"] + sh[:, 2])
        x["fb_divb"] = x["pfx_z"] - (x["fb_ivb_ref"] + sh[:, 0])
        x["fb_dhb"] = x["pfx_x"] - (x["fb_hb_ref"] + sh[:, 1])
    X = x[S["features"]]
    p = S["swing"].predict_proba(X)[:, 1] * S["whiff"].predict_proba(X)[:, 1]
    g = pd.Series((p - p0) * 100).groupby([x["pitcher"].values, x["pitch_type"].values]).mean()
    out = g.reindex(j.index)
    return out - out.groupby(level=1).transform("median")


labels = {("league", "v1"): "리그 무브 · v1 (지금 앱)", ("personal", "v1"): "개인 무브 · v1",
          ("league", "v2"): "리그 무브 · v2", ("personal", "v2"): "개인 무브 · v2 (1+2)",
          ("actual", "v1"): "실제 공 모양 · v1", ("actual", "v2"): "실제 공 모양 · v2"}
for (kind, ver) in labels:
    j[f"p_{kind}_{ver}"] = predict(ver, deltas[kind]).values

rows = []


def report(d, title):
    v_true = d["act"].var() - (d["se"] ** 2).mean()
    ceil = np.sqrt(max(v_true, 0) / d["act"].var())
    print(f"\n=== {title} (n={len(d)}, 최대 가능 상관 ≈{ceil:.2f})")
    for (kind, ver), lab in labels.items():
        x, y = d[f"p_{kind}_{ver}"].to_numpy(), d["act"].to_numpy()
        r, ci = corr_ci(x, y)
        slope = np.polyfit(x, y, 1)[0]
        print(f"  {lab:24s} {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  기울기 {slope:+.2f}")
        rows.append({"subset": title, "kind": kind, "ver": ver, "label": lab, "n": len(d), "r": r,
                     "lo": ci[0], "hi": ci[1], "slope": slope, "ceiling": ceil})


report(j, "전체")
report(j[j["moved"]], "릴리스 바꾼 경우")
grp = j.index.get_level_values("pitch_type").map(RM.GROUP)
report(j[j["moved"] & (grp == "fb")], "바꾼 경우 · 패스트볼")
report(j[j["moved"] & (grp != "fb")], "바꾼 경우 · 변화구+오프스피드")
report(j[grp != "fb"], "전체 · 변화구+오프스피드 (패스트볼 변화 영향 포함)")
pd.DataFrame(rows).to_csv("val_personal_summary.csv", index=False)
print("\n저장: val_personal_summary.csv")


# v2 − v1 차이가 우연인지: 같은 표본을 재추출해 상관 차이의 95% 구간
def diff_ci(d, c1, c2, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2, y = d[c1].to_numpy(), d[c2].to_numpy(), d["act"].to_numpy()
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        bs.append(np.corrcoef(x2[i], y[i])[0, 1] - np.corrcoef(x1[i], y[i])[0, 1])
    return np.corrcoef(x2, y)[0, 1] - np.corrcoef(x1, y)[0, 1], np.percentile(bs, [2.5, 97.5])


print("\n[차이 검정] 상관 차이 (95% 구간)")
DIFF = []
for sub, lab in ((j, "전체"), (j[grp != "fb"], "변화구+오프스피드 전체"), (j[j["moved"]], "릴리스 바꾼 경우")):
    for c1, c2, nm in (("p_actual_v1", "p_actual_v2", "공 모양: v2 − v1"),
                       ("p_league_v1", "p_personal_v1", "릴리스: 개인 − 리그 (v1)"),
                       ("p_league_v1", "p_league_v2", "릴리스: v2 − v1 (리그 무브)")):
        dlt, ci = diff_ci(sub, c1, c2)
        print(f"  {lab:18s} {nm:26s} {dlt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        DIFF.append({"subset": lab, "what": nm, "diff": dlt, "lo": ci[0], "hi": ci[1]})
pd.DataFrame(DIFF).to_csv("val_personal_diff.csv", index=False)
