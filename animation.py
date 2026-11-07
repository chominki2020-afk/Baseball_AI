"""
3D 투구 애니메이션: 투구 동작(막대 인형) + 공 비행

투구폼은 '데이터 기반 근사'예요.
  - 릴리스 순간 손 위치(높이·옆으로·익스텐션)와 팔 각도(arm angle)는 그 투수의 실제 스탯캐스트 평균값
  - 그 앞뒤 동작(세트 → 다리 들기 → 스트라이드 → 팔 젖히기 → 릴리스 → 팔로스루)은 일반적인 투구 동작을
    실제 릴리스 지점에 맞춰 이어 붙인 것 (관절 추적 데이터가 아님)

좌표: 스탯캐스트 기준(피트). x: 포수 시점 오른쪽(1루 쪽) +, y: 홈 끝 0 → 투수판 60.5, z: 높이
"""
import numpy as np
import plotly.graph_objects as go

import trajectory as T

RUBBER_Y = 60.5
BONES = [("neck", "head_base"), ("neck", "s_t"), ("neck", "s_g"), ("s_t", "e_t"), ("e_t", "h_t"),
         ("s_g", "e_g"), ("e_g", "h_g"), ("neck", "pelvis"), ("pelvis", "hip_t"), ("hip_t", "knee_t"),
         ("knee_t", "foot_t"), ("pelvis", "hip_g"), ("hip_g", "knee_g"), ("knee_g", "foot_g")]


def _norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def keyframes(rel_side, ext, rel_z, arm_deg, height=6.25):
    """투구 동작 키프레임 (로컬 좌표: l = 던지는 팔 쪽 +, f = 홈 쪽 +(투수판 기준), h = 높이)

    height: 투수 키(피트). 기본 동작은 키 6.25피트(190cm) 기준으로 만들고 높이 방향을 키에 맞춰 늘이고 줄임
    """
    sc = height / 6.25
    th = np.radians(arm_deg)
    R = np.array([rel_side, ext, rel_z])                       # 실제 릴리스 지점
    arm = 0.36 * height                                        # 어깨~손 길이 (키의 약 36%)
    S = R - arm * _norm([np.cos(th), 0.25, np.sin(th)])         # 던지는 어깨 = 릴리스에서 팔 각도 방향으로 역산
    G = S + np.array([-1.3, 0.2, 0.6 - 1.4 * np.sin(th)])       # 글러브 쪽 어깨 (오버핸드일수록 기울어짐)
    N = (S + G) / 2 + np.array([0, 0, 0.1])
    stride = max(ext - 1.0, 4.3)
    P = N + np.array([-0.15, -1.0, -1.9 * sc])
    P[2] = max(P[2], 2.6 * sc)
    rel = {"s_t": S, "s_g": G, "neck": N, "head_base": N + [-0.1, 0.3, 0.45],
           "e_t": S + 0.5 * (R - S) + [0, -0.1, 0.1], "h_t": R,
           "e_g": G + [-0.35, 0.45, -0.85], "h_g": N + [-0.55, 0.6, -0.65],
           "pelvis": P, "hip_t": P + [0.45, 0, 0], "hip_g": P + [-0.45, 0, 0],
           "knee_g": np.array([-0.45, stride - 0.2, 1.7]), "foot_g": np.array([-0.45, stride, 0.0]),
           "knee_t": np.array([0.45, (P[1] + 1.0) / 2, 1.0]), "foot_t": np.array([0.45, 1.0, 0.15])}

    def pose(d):
        return {k: np.asarray(v, float) for k, v in d.items()}

    k_set = pose({"neck": [0, 0.1, 5.1], "head_base": [0, 0.15, 5.4], "s_t": [0.1, -0.55, 5.0],
                  "s_g": [-0.1, 0.75, 5.0], "e_t": [0.45, -0.2, 4.4], "h_t": [0.3, 0.15, 4.3],
                  "e_g": [-0.2, 0.7, 4.3], "h_g": [0.2, 0.25, 4.3], "pelvis": [0, 0.2, 3.3],
                  "hip_t": [0.1, -0.2, 3.3], "hip_g": [-0.1, 0.6, 3.3], "knee_t": [0.2, -0.1, 1.7],
                  "knee_g": [-0.2, 0.9, 1.7], "foot_t": [0.3, 0.0, 0.0], "foot_g": [-0.3, 1.2, 0.0]})
    k_lift = dict(k_set)
    k_lift.update(pose({"knee_g": [-0.2, 1.3, 3.5], "foot_g": [-0.15, 0.8, 2.1], "hip_g": [-0.1, 0.5, 3.35],
                        "h_t": [0.3, 0.2, 4.7], "h_g": [0.2, 0.3, 4.7], "e_t": [0.5, -0.2, 4.7],
                        "e_g": [-0.2, 0.7, 4.6]}))
    k_stride = pose({"neck": [0.05, 2.5, 5.0], "head_base": [-0.05, 2.7, 5.3], "s_t": [0.25, 1.9, 4.85],
                     "s_g": [-0.15, 3.1, 4.9], "e_t": [0.9, 1.4, 4.2], "h_t": [1.1, 0.6, 3.6],
                     "e_g": [-0.45, 3.6, 4.9], "h_g": [-0.45, 4.3, 4.7], "pelvis": [0, 2.4, 3.0],
                     "hip_t": [0.3, 2.1, 3.0], "hip_g": [-0.3, 2.7, 3.0], "knee_t": [0.4, 1.3, 1.6],
                     "knee_g": [-0.35, stride - 1.5, 2.0], "foot_t": [0.45, 0.4, 0.1],
                     "foot_g": [-0.4, stride - 0.6, 0.45]})
    back = np.array([0, -0.9, 0])
    k_cock = pose({"neck": N + back, "head_base": N + back + [-0.1, 0.2, 0.45], "s_t": S + back,
                   "s_g": G + back + [0, 0.2, 0.2], "e_t": S + back + [0.85, -0.1, 0.15],
                   "h_t": S + back + [0.7, -0.6, 0.9], "e_g": G + back + [-0.3, 0.8, -0.3],
                   "h_g": G + back + [-0.4, 1.3, -0.6], "pelvis": P + [0, -0.5, 0.15],
                   "hip_t": P + [0.45, -0.5, 0.15], "hip_g": P + [-0.45, -0.5, 0.15],
                   "knee_g": [-0.45, stride - 0.3, 1.6], "foot_g": [-0.45, stride, 0.0],
                   "knee_t": [0.45, 1.9, 1.1], "foot_t": [0.45, 1.0, 0.15]})
    k_rel = pose(rel)
    fwd = np.array([0, 0.7, -0.5])
    k_follow = pose({"neck": N + fwd, "head_base": N + fwd + [-0.2, 0.35, 0.35], "s_t": S + fwd + [-0.4, 0, 0],
                     "s_g": G + fwd + [0.2, -0.3, -0.2], "e_t": N + fwd + [-0.4, 0.6, -0.9],
                     "h_t": [-1.0, ext + 0.4, 2.3], "e_g": G + fwd + [-0.2, -0.3, -0.9],
                     "h_g": G + fwd + [-0.1, -0.6, -1.4], "pelvis": P + [0, 0.6, -0.2],
                     "hip_t": P + [0.4, 0.6, -0.2], "hip_g": P + [-0.4, 0.6, -0.2],
                     "knee_g": [-0.45, stride - 0.1, 1.55], "foot_g": [-0.45, stride, 0.0],
                     "knee_t": [0.35, stride - 1.4, 1.9], "foot_t": [0.55, stride - 2.2, 1.3]})
    k_end = dict(k_follow)
    k_end.update(pose({"knee_t": [0.4, stride - 0.4, 1.4], "foot_t": [0.6, stride - 0.6, 0.3]}))
    # 키에 맞추기: 정해진 숫자로 만든 관절(세트·다리 들기·스트라이드 전체, 나머지 동작의 다리)은 높이를 키 비율만큼
    for k in (k_set, k_lift, k_stride):
        for j in k:
            k[j] = k[j] * [1, 1, sc]
    for k in (k_cock, k_rel, k_follow, k_end):
        for j in ("knee_g", "foot_g", "knee_t", "foot_t"):
            k[j] = k[j] * [1, 1, sc]
    k_follow["h_t"] = k_follow["h_t"] * [1, 1, sc]
    k_end["h_t"] = k_follow["h_t"]
    return [(-1.15, k_set), (-0.8, k_lift), (-0.38, k_stride), (-0.12, k_cock), (0.0, k_rel),
            (0.32, k_follow), (0.75, k_end)]


def pose_at(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)                              # 부드럽게 (smoothstep)
            return {k: a[k] + (b[k] - a[k]) * u for k in a}
    return keys[-1][1]


# ---------------------------------------------------------------- 실제 동작 (OpenBiomechanics)
def load_library(path="obp_motions.npz"):
    """build_motion.py가 만든 실제 투구 동작 100개 (Driveline OBP, CC BY-NC-SA 4.0)"""
    import json
    import os
    import pandas as pd
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {"motions": z["motions"], "grid": z["grid"], "names": [str(n) for n in z["names"]],
            "info": pd.DataFrame(json.loads(str(z["info"])))}


def pick_motion(lib, arm_deg, height_ft):
    """팔 각도가 가장 비슷한 실제 투수 (키는 약간만 고려)"""
    info = lib["info"]
    score = (info["arm_angle"] - arm_deg).abs() + 2.0 * (info["height_ft"] - height_ft).abs()
    return int(score.idxmin())


def fit_motion(lib, idx, rel_side, ext, rel_z, height_ft):
    """실제 동작을 이 투수의 키·릴리스 위치에 맞추기
       - 앞뒤·좌우: 키 비율로 확대/축소한 뒤, 릴리스 순간 손이 실제 릴리스 지점에 오도록 이동
       - 높이: 발은 땅에 붙인 채, 릴리스 순간 손 높이가 실제 타점과 같아지도록 늘이고 줄임"""
    m = lib["motions"][idx].astype(float)
    names, grid = lib["names"], lib["grid"]
    i0 = int(np.argmin(np.abs(grid)))
    hand = m[i0, names.index("h_t")]
    s = height_ft / lib["info"].loc[idx, "height_ft"]
    sz = rel_z / hand[2]
    out = m.copy()
    out[..., 0] = m[..., 0] * s + (rel_side - hand[0] * s)
    out[..., 1] = m[..., 1] * s + (ext - hand[1] * s)
    out[..., 2] = m[..., 2] * sz
    keys = {"grid": grid, "arr": out, "names": names}
    return keys


def motion_pose(mot, t):
    g = mot["grid"]
    t = min(max(t, g[0]), g[-1])
    k = int(np.searchsorted(g, t))
    k = min(max(k, 1), len(g) - 1)
    u = (t - g[k - 1]) / (g[k] - g[k - 1])
    a = mot["arr"][k - 1] + (mot["arr"][k] - mot["arr"][k - 1]) * u
    return {n: a[j] for j, n in enumerate(mot["names"])}


def to_world(p, side_sign):
    """로컬(l, f, h) → 스탯캐스트 (x, y, z). side_sign: 우완 -1(3루 쪽), 좌완 +1"""
    return np.array([side_sign * p[0], RUBBER_Y - p[1], p[2]])


def figure_trace(pose, side_sign):
    xs, ys, zs = [], [], []
    for a, b in BONES:
        pa, pb = to_world(pose[a], side_sign), to_world(pose[b], side_sign)
        xs += [pa[0], pb[0], None]
        ys += [pa[1], pb[1], None]
        zs += [pa[2], pb[2], None]
    head = to_world(pose["head_base"] + np.array([0, 0, 0.35]), side_sign)
    return (xs, ys, zs), head


def batter_traces(stand, zone_top):
    """타석의 타자 (정지된 막대 인형, 배트 포함)"""
    s = -1 if stand == "R" else 1                                # 우타자는 3루 쪽(x -)
    bx = s * 2.4
    pts = {"head": [bx, 0.4, 5.6], "neck": [bx, 0.4, 5.1], "pelvis": [bx, 0.5, 3.2],
           "kf": [bx - s * 0.1, 1.4, 1.7], "ff": [bx, 2.0, 0], "kb": [bx + s * 0.2, -0.5, 1.7], "fb": [bx + s * 0.1, -1.0, 0],
           "hand": [bx + s * 0.35, -0.3, 4.8], "bat": [bx + s * 0.9, -1.6, 6.6]}
    segs = [("neck", "pelvis"), ("pelvis", "kf"), ("kf", "ff"), ("pelvis", "kb"), ("kb", "fb"), ("neck", "hand")]
    xs, ys, zs = [], [], []
    for a, b in segs:
        xs += [pts[a][0], pts[b][0], None]
        ys += [pts[a][1], pts[b][1], None]
        zs += [pts[a][2], pts[b][2], None]
    body = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#9aa0a6", width=7),
                        hoverinfo="skip", showlegend=False)
    bat = go.Scatter3d(x=[pts["hand"][0], pts["bat"][0]], y=[pts["hand"][1], pts["bat"][1]],
                       z=[pts["hand"][2], pts["bat"][2]], mode="lines", line=dict(color="#8d6e63", width=9),
                       hoverinfo="skip", showlegend=False)
    head = go.Scatter3d(x=[pts["head"][0]], y=[pts["head"][1]], z=[pts["head"][2]], mode="markers",
                        marker=dict(size=11, color="#9aa0a6"), hoverinfo="skip", showlegend=False)
    return [body, bat, head]


CAMERAS = {
    "투수 뒤": dict(eye=dict(x=0.35, y=3.1, z=0.45), center=dict(x=0, y=0.6, z=-0.05)),
    "타자 시점": dict(eye=dict(x=0.0, y=-2.6, z=0.2), center=dict(x=0, y=0.0, z=-0.05)),
    "옆에서": dict(eye=dict(x=3.0, y=0.0, z=0.3), center=dict(x=0, y=0, z=-0.1)),
    "위에서": dict(eye=dict(x=0.01, y=0.0, z=3.2), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
}


def make_animation(tr_a, tr_b, pa, arm_deg, stand, zone_bot, zone_top, color_a, color_b, label_a, label_b,
                   height=6.25, motion=None):
    """motion: fit_motion()으로 맞춘 실제 동작. 없으면 만든 동작(keyframes) 사용"""
    side_sign = -1 if pa["release_pos_x"] < 0 else 1
    keys = keyframes(abs(pa["release_pos_x"]), RUBBER_Y - pa["release_pos_y"], pa["release_pos_z"], arm_deg, height)
    t_end = max(tr_a[0][-1], tr_b[0][-1] if tr_b is not None else 0) + 0.1
    times = list(np.arange(-1.15, 0, 0.05)) + list(np.arange(0, t_end + 1e-9, 0.016))   # 비행 구간은 촘촘히(슬로모션)

    def ball(tr, t, pose):
        if tr is None or t < 0:
            h = to_world(pose["h_t"], side_sign)
            return [h[0]], [h[1]], [h[2]]
        x, y, z = T.at_time(tr, min(t, tr[0][-1]))
        return [x], [y], [z]

    def trail(tr, t):
        if tr is None or t <= 0:
            return [None], [None], [None]
        m = tr[0] <= t
        return list(tr[1][m]), list(tr[2][m]), list(tr[3][m])

    def dyn(t):
        pose = motion_pose(motion, t) if motion is not None else pose_at(keys, t)
        (fx, fy, fz), head = figure_trace(pose, side_sign)
        bx, by, bz = ball(tr_a, t, pose)
        cx, cy, cz = ball(tr_b, t, pose) if tr_b is not None else ([None], [None], [None])
        ax_, ay_, az_ = trail(tr_a, t)
        qx, qy, qz = trail(tr_b, t)
        return [
            go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="#3c4043", width=8),
                         hoverinfo="skip", showlegend=False),
            go.Scatter3d(x=[head[0]], y=[head[1]], z=[head[2]], mode="markers",
                         marker=dict(size=12, color="#3c4043"), hoverinfo="skip", showlegend=False),
            go.Scatter3d(x=ax_, y=ay_, z=az_, mode="lines", line=dict(color=color_a, width=7), name=label_a),
            go.Scatter3d(x=qx, y=qy, z=qz, mode="lines", line=dict(color=color_b, width=7), name=label_b,
                         visible=tr_b is not None),
            go.Scatter3d(x=bx, y=by, z=bz, mode="markers", showlegend=False,
                         marker=dict(size=7, color="white", line=dict(color=color_a, width=4))),
            go.Scatter3d(x=cx, y=cy, z=cz, mode="markers", showlegend=False, visible=tr_b is not None,
                         marker=dict(size=7, color="white", opacity=0.75, line=dict(color=color_b, width=4))),
        ]

    # 고정된 배경: 스트라이크존, 홈플레이트, 땅, 마운드, 투수판, 타자
    statics = []
    zx = [-0.83, 0.83, 0.83, -0.83, -0.83]
    statics.append(go.Scatter3d(x=zx, y=[T.PLATE_Y] * 5, z=[zone_bot, zone_bot, zone_top, zone_top, zone_bot],
                                mode="lines", line=dict(color="#3c4043", width=5), name="스트라이크존"))
    statics.append(go.Scatter3d(x=[-0.71, 0.71, 0.71, 0, -0.71, -0.71], y=[1.417, 1.417, 0.708, 0, 0.708, 1.417],
                                z=[0] * 6, mode="lines", line=dict(color="#9aa0a6", width=4),
                                hoverinfo="skip", showlegend=False))
    statics.append(go.Mesh3d(x=[-4, 4, 4, -4], y=[-3, -3, 64, 64], z=[0] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                             color="#c8a27a", opacity=0.22, hoverinfo="skip", showlegend=False))
    ang = np.linspace(0, 2 * np.pi, 40)
    statics.append(go.Scatter3d(x=9 * np.cos(ang), y=59 + 9 * np.sin(ang), z=[0.02] * 40, mode="lines",
                                line=dict(color="#b08960", width=3), hoverinfo="skip", showlegend=False))
    statics.append(go.Scatter3d(x=[-1, 1], y=[RUBBER_Y, RUBBER_Y], z=[0.05, 0.05], mode="lines",
                                line=dict(color="white", width=8), hoverinfo="skip", showlegend=False))
    for tr, c in ((tr_a, color_a), (tr_b, color_b)):            # 타자 결정 시점 표시
        if tr is not None:
            xd, yd, zd = T.at_time(tr, tr[0][-1] - T.DECISION_T)
            statics.append(go.Scatter3d(x=[xd], y=[yd], z=[zd], mode="markers", showlegend=False,
                                        marker=dict(size=4, color=c, symbol="diamond"), hoverinfo="skip"))
    statics += batter_traces(stand, zone_top)

    first = dyn(times[0])
    fig = go.Figure(data=first + statics)
    n_dyn = len(first)
    frames = []
    for i, t in enumerate(times):
        label = f"{t:+.2f}초" if t < 0 else f"릴리스 후 {t:.2f}초"
        frames.append(go.Frame(data=dyn(t), traces=list(range(n_dyn)), name=str(i),
                               layout=go.Layout(title_text=label)))
    fig.frames = frames

    play_slow = dict(label="▶ 재생 (공은 슬로모션)", method="animate",
                     args=[None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": False,
                                  "transition": {"duration": 0}, "mode": "immediate"}])
    pause = dict(label="■ 멈춤", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
    cams = [dict(label=k, method="relayout", args=[{"scene.camera": v}]) for k, v in CAMERAS.items()]
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=40, b=0), title=dict(text="▶ 재생을 눌러 보세요", x=0.5, font=dict(size=14)),
        legend=dict(x=0, y=0.92),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.06, xanchor="left", showactive=False,
                          buttons=[play_slow, pause]),
                     dict(type="buttons", direction="left", x=0.0, y=0.1, xanchor="left", yanchor="bottom",
                          showactive=True, buttons=cams)],          # 카메라 버튼은 아래쪽 (위쪽은 확대/회전 도구와 겹침)
        sliders=[dict(active=0, x=0.05, len=0.9, y=0.02, currentvalue=dict(visible=False),
                      steps=[dict(method="animate", label="", args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                                                              "mode": "immediate"}])
                             for i in range(len(times))])],
        scene=dict(xaxis=dict(range=[-5, 5], visible=False), yaxis=dict(range=[-4, 64], visible=False),
                   zaxis=dict(range=[0, 8.5], visible=False), aspectmode="manual",
                   aspectratio=dict(x=1, y=4.2, z=0.85), camera=CAMERAS["투수 뒤"]))
    return fig
