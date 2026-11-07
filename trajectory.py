"""
투구 궤적 계산

스탯캐스트는 공마다 '홈에서 50피트 지점의 속도(vx0, vy0, vz0)'와 '가속도(ax, ay, az, 중력+공기저항+회전 효과)'를 줌.
투수·구종별 평균값으로, 원하는 목표 코스에 도착하도록 처음 던지는 방향만 조절해서 궤적을 그림
(가속도 = 그 투수 그 구종의 '휘는 성질'은 그대로 유지).

좌표 (스탯캐스트 기준, 단위 피트)
  x: 포수 시점에서 오른쪽(+) = 1루 쪽,  y: 홈플레이트 끝 0 → 투수 쪽(+),  z: 높이

  python build_physics.py 로 physics.pkl을 먼저 만들어야 함
"""
import numpy as np

PLATE_Y = 17 / 12          # 홈플레이트 앞면
DECISION_T = 0.175         # 타자가 스윙을 결정해야 하는 시점: 도착 약 0.175초 전


def path(p, target_x, target_z, n=80):
    """p = 투수·구종 평균 물리값(dict). 목표 (plate_x, plate_z)에 도착하는 궤적 → (t, x, y, z) 배열"""
    ax, ay, az = p["ax"], p["ay"], p["az"]
    x0, y0, z0 = p["release_pos_x"], p["release_pos_y"], p["release_pos_z"]
    # 50피트 지점 속도 → 릴리스 지점 속도 (y 방향은 등가속도 공식 v² = v0² + 2aΔy)
    vy_r = -np.sqrt(max(p["vy0"] ** 2 - 2 * ay * (50 - y0), 1.0))
    # 홈까지 걸리는 시간: y0 + vy t + ½ ay t² = PLATE_Y
    a, b, c = 0.5 * ay, vy_r, y0 - PLATE_Y
    t_end = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a) if abs(a) > 1e-9 else -c / b
    if t_end <= 0:
        t_end = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    # 목표 지점에 맞도록 x, z 방향 처음 속도를 역산
    vx_r = (target_x - x0 - 0.5 * ax * t_end ** 2) / t_end
    vz_r = (target_z - z0 - 0.5 * az * t_end ** 2) / t_end
    t = np.linspace(0, t_end, n)
    x = x0 + vx_r * t + 0.5 * ax * t ** 2
    y = y0 + vy_r * t + 0.5 * ay * t ** 2
    z = z0 + vz_r * t + 0.5 * az * t ** 2
    return t, x, y, z


def at_time(tr, t_query):
    t, x, y, z = tr
    return np.interp(t_query, t, x), np.interp(t_query, t, y), np.interp(t_query, t, z)


def straight_line(p, target_x, target_z, n=80):
    """비교용: 회전·공기 효과 없이 중력만 받았다면 (= '휘지 않는 공')"""
    q = dict(p)
    q["ax"], q["az"] = 0.0, -32.17
    return path(q, target_x, target_z, n)


def tunnel(tr_a, tr_b):
    """두 공이 '결정 시점'에 얼마나 붙어 있었나 vs 홈에서 얼마나 벌어졌나 (인치)"""
    ta, tb = tr_a[0][-1], tr_b[0][-1]
    xa, ya, za = at_time(tr_a, ta - DECISION_T)
    xb, yb, zb = at_time(tr_b, tb - DECISION_T)
    gap_decide = np.hypot(xa - xb, za - zb) * 12
    gap_plate = np.hypot(tr_a[1][-1] - tr_b[1][-1], tr_a[3][-1] - tr_b[3][-1]) * 12
    return gap_decide, gap_plate, (ya + yb) / 2


def approach_angles(tr):
    """홈에 들어올 때 각도 (도). VAA: 수직 진입각(-면 내려꽂힘, 0에 가까울수록 평평), HAA: 좌우 진입각"""
    t, x, y, z = tr
    dt = t[-1] - t[-2]
    vx, vy, vz = (x[-1] - x[-2]) / dt, (y[-1] - y[-2]) / dt, (z[-1] - z[-2]) / dt
    return np.degrees(np.arctan2(vz, -vy)), np.degrees(np.arctan2(vx, -vy))


def same_tunnel_target(p, tr_a):
    """B 구종을 '타자 결정 시점에 A와 같은 위치'를 지나가게 던지면 홈 어디에 도착하나 → 그 목표 (x, z)

    궤적 위치는 목표 지점에 대해 1차식이라 바로 풀림:
      pos(td) = start·(1-k) + target·k + ½a(td² - T·td),   k = td / T
    """
    xa, _, za = at_time(tr_a, tr_a[0][-1] - DECISION_T)
    probe = path(p, 0.0, 2.5)                       # 비행 시간 T만 알면 됨 (목표와 무관)
    T = probe[0][-1]
    td = T - DECISION_T
    k = td / T
    x0, z0 = p["release_pos_x"], p["release_pos_z"]
    tx = (xa - x0 * (1 - k) - 0.5 * p["ax"] * (td ** 2 - T * td)) / k
    tz = (za - z0 * (1 - k) - 0.5 * p["az"] * (td ** 2 - T * td)) / k
    return tx, tz


def with_release(p, d_height=0.0, d_side=0.0, d_ext=0.0):
    """릴리스 포인트 바꿔 보기: 높이(+면 높게), 좌우(+면 몸 바깥쪽으로 더 벌림 = 사이드암 쪽), 익스텐션(+면 홈에 더 가까이)"""
    q = dict(p)
    arm = -1 if q["release_pos_x"] < 0 else 1          # 우완은 x가 -(3루 쪽)
    q["release_pos_z"] = p["release_pos_z"] + d_height
    q["release_pos_x"] = p["release_pos_x"] + arm * d_side
    q["release_pos_y"] = p["release_pos_y"] - d_ext
    return q


def tunnel_words(gap):
    if gap < 3:
        return "거의 구분 불가"
    if gap < 6:
        return "구분 어려움"
    if gap < 10:
        return "구분 가능"
    return "구분 쉬움"
