"""
선수 임베딩 (딥러닝) — 투구 수십만 개에서 타자·투수마다 '성향 벡터'를 신경망이 스스로 학습

모델
  공 정보(구속·무브·코스·카운트·상황)  → MLP → h
  타자 ID → 임베딩 b (16차원),  투수 ID → 임베딩 p (16차원)
  [h, b, p, h⊙b, h⊙p] → MLP → 스윙 확률 / 헛스윙 확률(스윙했을 때) / 실점 가치
  - 임베딩에 L2 벌점 → 표본 적은 선수는 0(= 리그 평균) 쪽으로 자연히 당겨짐 (지금의 '축소 보정'을 신경망이 대신)
  - h⊙b : 같은 타자라도 공 종류·코스에 따라 반응이 달라지는 걸 표현 (예: 높은 직구엔 강한데 낮은 변화구엔 약함)

검증 (지금 방식과 같은 기준으로 비교)
  2024로 학습 → 2025 예측.  비교 대상 = 기존 LightGBM + 타자 프로필(축소 보정) 모델
  타자 2024 표본 크기별로도 비교 (표본 적은 선수에서 임베딩이 더 나은가?)

실행: python build_embed.py val    (검증, 15~25분)
      python build_embed.py save   (2024+2025로 최종 학습 → embed_model.npz, 앱은 numpy만으로 사용)
"""
import sys
import time

import numpy as np
import pandas as pd

import build_matchup as B

FILES = {2024: "statcast_2024-04-01_2024-09-30.csv", 2025: "statcast_2025-03-27_2025-09-28.csv"}
CONT = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate", "x_in", "z_norm", "vaa_rel", "balls", "strikes",
        "same_hand", "outs", "r1", "r2", "r3"]
EMB = 16
H = 64


def xwcon(df):
    """인플레이 타구의 xwOBA (나머지는 NaN)"""
    return np.where(df["in_play"] == 1, df["estimated_woba_using_speedangle"], np.nan)


def load(year):
    df = B.load(FILES[year])
    df["season"] = year
    return df


def prep(df, scaler=None):
    X = df[CONT].astype(float)
    if scaler is None:
        scaler = {"mu": X.mean().to_numpy(), "sd": X.std().replace(0, 1).to_numpy()}
    Z = ((X.to_numpy() - scaler["mu"]) / scaler["sd"])
    Z = np.nan_to_num(Z, nan=0.0)
    oh = np.eye(3)[df["grp"].astype(int).clip(0, 2).to_numpy()]
    return np.hstack([Z, oh]).astype(np.float32), scaler


def id_map(series, min_n=1):
    vc = series.value_counts()
    ids = vc[vc >= min_n].index.to_numpy()
    return {int(k): i + 1 for i, k in enumerate(ids)}      # 0 = 모르는 선수 (리그 평균)


def to_idx(series, mp):
    return series.map(mp).fillna(0).astype(int).to_numpy()


# ------------------------------------------------------------------ PyTorch 모델 (학습할 때만 필요)
def make_model(nb, npch, d_in):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.eb = nn.Embedding(nb, EMB)
            s.ep = nn.Embedding(npch, EMB)
            nn.init.normal_(s.eb.weight, 0, 0.01)
            nn.init.normal_(s.ep.weight, 0, 0.01)
            s.f = nn.Sequential(nn.Linear(d_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            s.proj = nn.Linear(H, EMB)
            s.g = nn.Sequential(nn.Linear(H + 4 * EMB, H), nn.ReLU(), nn.Linear(H, 4))

        def forward(s, x, bi, pi):
            h = s.f(x)
            b, p = s.eb(bi), s.ep(pi)
            q = s.proj(h)
            z = torch.cat([h, b, p, q * b, q * p], 1)
            return s.g(z)          # [스윙 로짓, 헛스윙 로짓, 실점 가치, 인플레이 xwOBA(타구 질 — 파워 성향을 임베딩에 담기)]
    return Net()


def train(X, bi, pi, swing, whiff, rv, nb, npch, epochs=6, lam=5.0, seed=0, log=True, xw=None):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = make_model(nb, npch, X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt, bt, pt = torch.tensor(X), torch.tensor(bi), torch.tensor(pi)
    st, wt = torch.tensor(swing, dtype=torch.float32), torch.tensor(whiff, dtype=torch.float32)
    rv_sd = float(np.nanstd(rv))
    rt = torch.tensor(np.nan_to_num(rv / rv_sd), dtype=torch.float32)
    xw = np.full(len(X), np.nan) if xw is None else np.asarray(xw, float)
    xw_mu, xw_sd = float(np.nanmean(xw)) if np.isfinite(xw).any() else 0.0, float(np.nanstd(xw)) if np.isfinite(xw).any() else 1.0
    xm = torch.tensor(np.isfinite(xw))
    xt = torch.tensor(np.nan_to_num((xw - xw_mu) / xw_sd), dtype=torch.float32)
    n = len(X)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for k in range(0, n, 4096):
            i = perm[k:k + 4096]
            out = net(Xt[i], bt[i], pt[i])
            ls = bce(out[:, 0], st[i]).mean()
            m = st[i] > 0.5
            lw = bce(out[m, 1], wt[i][m]).mean() if m.any() else 0.0
            lr_ = ((out[:, 2] - rt[i]) ** 2).mean()
            mx = xm[i]
            lx = ((out[mx, 3] - xt[i][mx]) ** 2).mean() if mx.any() else 0.0
            # MAP 축소: 전체 손실 = Σ투구 손실 + λ Σ선수 ||임베딩||²  → 배치 평균 기준으로 λ/n
            emb_pen = lam / n * (net.eb.weight.pow(2).sum() + net.ep.weight.pow(2).sum())
            loss = ls + lw + 0.3 * lr_ + 0.5 * lx + emb_pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(i)
        if log:
            print(f"  epoch {ep + 1}/{epochs}  loss {tot / n:.4f}  ({time.time() - t0:.0f}s)")
    net.rv_sd, net.xw_mu, net.xw_sd = rv_sd, xw_mu, xw_sd
    return net


def export(net, scaler, bmap, pmap, extra=None):
    """numpy 배열로 내보내기 → 앱(윈도우)에서는 torch 없이 numpy만으로 계산"""
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    out = {"eb": sd["eb.weight"], "ep": sd["ep.weight"], "proj_w": sd["proj.weight"], "proj_b": sd["proj.bias"],
           "f0_w": sd["f.0.weight"], "f0_b": sd["f.0.bias"], "f2_w": sd["f.2.weight"], "f2_b": sd["f.2.bias"],
           "g0_w": sd["g.0.weight"], "g0_b": sd["g.0.bias"], "g2_w": sd["g.2.weight"], "g2_b": sd["g.2.bias"],
           "mu": scaler["mu"], "sd": scaler["sd"], "rv_sd": np.array(net.rv_sd),
           "xw_mu": np.array(net.xw_mu), "xw_sd": np.array(net.xw_sd),
           "b_ids": np.array(list(bmap.keys())), "b_idx": np.array(list(bmap.values())),
           "p_ids": np.array(list(pmap.keys())), "p_idx": np.array(list(pmap.values()))}
    out.update(extra or {})
    return out


def np_forward(W, X, bi, pi):
    relu = lambda a: np.maximum(a, 0)
    h = relu(relu(X @ W["f0_w"].T + W["f0_b"]) @ W["f2_w"].T + W["f2_b"])
    b, p = W["eb"][bi], W["ep"][pi]
    q = h @ W["proj_w"].T + W["proj_b"]
    z = np.hstack([h, b, p, q * b, q * p])
    o = relu(z @ W["g0_w"].T + W["g0_b"]) @ W["g2_w"].T + W["g2_b"]
    sig = lambda a: 1 / (1 + np.exp(-a))
    return sig(o[:, 0]), sig(o[:, 1]), o[:, 2] * float(W["rv_sd"])


def np_forward_xw(W, X, bi, pi):
    """인플레이 xwOBA 예측 (타구 질)"""
    relu = lambda a: np.maximum(a, 0)
    h = relu(relu(X @ W["f0_w"].T + W["f0_b"]) @ W["f2_w"].T + W["f2_b"])
    b, p = W["eb"][bi], W["ep"][pi]
    q = h @ W["proj_w"].T + W["proj_b"]
    o = relu(np.hstack([h, b, p, q * b, q * p]) @ W["g0_w"].T + W["g0_b"]) @ W["g2_w"].T + W["g2_b"]
    return o[:, 3] * float(W["xw_sd"]) + float(W["xw_mu"])


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return roc_auc_score(y, p), log_loss(y, p)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "val"
    import build_swing as W_
    d24 = load(2024)
    if step == "tune":                 # 하이퍼파라미터는 2024 안에서만 고름 (4~7월 학습 → 8~9월 확인) — 2025는 건드리지 않음
        a, b = d24[d24["game_date"] < "2024-08-01"], d24[d24["game_date"] >= "2024-08-01"]
        Xa, sc = prep(a)
        Xb, _ = prep(b, sc)
        bm, pm = id_map(a["batter"]), id_map(a["pitcher"])
        for ep, lam in ((6, 5.0), (15, 5.0), (15, 20.0), (25, 20.0), (25, 60.0)):
            net = train(Xa, to_idx(a["batter"], bm), to_idx(a["pitcher"], pm), a["swing"].to_numpy(float),
                        a["whiff"].to_numpy(float), a[B.TARGET].to_numpy(float), len(bm) + 1, len(pm) + 1,
                        epochs=ep, lam=lam, log=False, xw=xwcon(a))
            ps, pw, _ = np_forward(export(net, sc, bm, pm), Xb, to_idx(b["batter"], bm), to_idx(b["pitcher"], pm))
            sw = b["swing"] == 1
            print(f"  epochs {ep:>2} λ {lam:>4}: 스윙 AUC {metrics(b['swing'], ps)[0]:.4f} ll {metrics(b['swing'], ps)[1]:.4f} | "
                  f"헛스윙 AUC {metrics(b[sw]['whiff'], pw[sw.to_numpy()])[0]:.4f} ll {metrics(b[sw]['whiff'], pw[sw.to_numpy()])[1]:.4f}")
        sys.exit()

    if step == "val":
        d25 = load(2025)
        # --- 신경망: 2024 전체로 학습
        X24, sc = prep(d24)
        bmap, pmap = id_map(d24["batter"]), id_map(d24["pitcher"])
        print(f"타자 {len(bmap)}명, 투수 {len(pmap)}명, 투구 {len(d24):,}개")
        net = train(X24, to_idx(d24["batter"], bmap), to_idx(d24["pitcher"], pmap), d24["swing"].to_numpy(float),
                    d24["whiff"].to_numpy(float), d24[B.TARGET].to_numpy(float), len(bmap) + 1, len(pmap) + 1,
                    xw=xwcon(d24))
        Wn = export(net, sc, bmap, pmap)
        X25, _ = prep(d25, sc)
        ps, pw, _ = np_forward(Wn, X25, to_idx(d25["batter"], bmap), to_idx(d25["pitcher"], pmap))
        # 임베딩 없이 (모든 선수 0 = 리그 평균) — 임베딩이 얼마나 기여하는지
        z = np.zeros(len(d25), int)
        ps0, pw0, _ = np_forward(Wn, X25, z, z)
        # --- 기존 방식: LightGBM + 타자 프로필 (전반기 프로필 → 후반기 학습, 2025엔 2024 전체 프로필)
        first, second = d24[d24["game_date"] < "2024-07-01"], d24[d24["game_date"] >= "2024-07-01"]
        tr = B.attach(second, B.batter_profile(first))
        te = B.attach(d25, B.batter_profile(d24))
        F = W_.FULL_F
        g_sw = W_.clf().fit(tr[F], tr["swing"])
        g_wh = W_.clf().fit(tr[tr["swing"] == 1][F], tr[tr["swing"] == 1]["whiff"])
        gs, gw = g_sw.predict_proba(te[F])[:, 1], g_wh.predict_proba(te[F])[:, 1]
        # --- 비교
        n24 = d24["batter"].value_counts()
        d25 = d25.assign(n24=d25["batter"].map(n24).fillna(0).to_numpy(), ps=ps, pw=pw, ps0=ps0, pw0=pw0, gs=gs, gw=gw)
        d25["avg"] = (d25["ps"] + d25["gs"]) / 2
        d25["avgw"] = (d25["pw"] + d25["gw"]) / 2
        rows = []
        buckets = [("전체", d25), ("2024 표본 0 (신인)", d25[d25["n24"] == 0]),
                   ("2024 1~500구", d25[(d25["n24"] > 0) & (d25["n24"] <= 500)]),
                   ("2024 500~1500구", d25[(d25["n24"] > 500) & (d25["n24"] <= 1500)]),
                   ("2024 1500구+", d25[d25["n24"] > 1500])]
        print("\n[2024 학습 → 2025 예측]  AUC↑ / logloss↓")
        for lab, d in buckets:
            sw = d[d["swing"] == 1]
            line = f"  {lab:16s} n={len(d):>7,} |"
            for nm, cs, cw in (("LightGBM+프로필", "gs", "gw"), ("신경망+임베딩", "ps", "pw"),
                               ("신경망(임베딩 끔)", "ps0", "pw0"), ("둘 평균", "avg", "avgw")):
                a1, l1 = metrics(d["swing"], d[cs])
                a2, l2 = metrics(sw["whiff"], sw[cw])
                line += f" {nm} 스윙 {a1:.4f}/{l1:.4f} 헛스윙 {a2:.4f}/{l2:.4f} |"
                rows.append({"bucket": lab, "n": len(d), "model": nm, "swing_auc": a1, "swing_ll": l1,
                             "whiff_auc": a2, "whiff_ll": l2})
            print(line)
        pd.DataFrame(rows).to_csv("val_embed_summary.csv", index=False)
        np.savez_compressed("embed_val2024.npz", **Wn)
        print("저장: val_embed_summary.csv")

    if step == "save":
        d25 = load(2025)
        both = pd.concat([d24, d25], ignore_index=True)
        X, sc = prep(both)
        bmap, pmap = id_map(both["batter"]), id_map(both["pitcher"])
        net = train(X, to_idx(both["batter"], bmap), to_idx(both["pitcher"], pmap), both["swing"].to_numpy(float),
                    both["whiff"].to_numpy(float), both[B.TARGET].to_numpy(float), len(bmap) + 1, len(pmap) + 1,
                    epochs=7, xw=xwcon(both))
        # 선수별 표본 수·이름·기본 지표 (지도 해석용)
        nb = both["batter"].value_counts()
        npc = both["pitcher"].value_counts()
        g = both.groupby("batter")
        bstat = pd.DataFrame({"n": nb, "swing": g["swing"].mean(), "whiff": both[both["swing"] == 1].groupby("batter")["whiff"].mean(),
                              "chase": both[both["oz"] > 0].groupby("batter")["swing"].mean(),
                              "zswing": both[both["oz"] == 0].groupby("batter")["swing"].mean(),
                              "rv100": g[B.TARGET].mean() * 100,
                              "ev": both[both["in_play"] == 1].groupby("batter")["launch_speed"].mean()})
        gp = both.groupby("pitcher")
        pstat = pd.DataFrame({"n": npc, "name": gp["player_name"].first(), "hand": gp["p_throws"].first(),
                              "velo": both[both["grp"] == 0].groupby("pitcher")["release_speed"].mean(),
                              "whiff": both[both["swing"] == 1].groupby("pitcher")["whiff"].mean(),
                              "rv100": gp[B.TARGET].mean() * 100,
                              "fb_share": gp["grp"].apply(lambda s: (s == 0).mean())})
        stand = both.groupby("batter")["stand"].agg(lambda s: "S" if s.nunique() > 1 else s.iloc[0])
        bstat["stand"] = stand
        Wn = export(net, sc, bmap, pmap, extra={
            "bstat_cols": np.array(list(bstat.columns)), "bstat_ids": bstat.index.to_numpy(),
            **{f"bstat_{c}": bstat[c].to_numpy() for c in bstat.columns},
            "pstat_cols": np.array(list(pstat.columns)), "pstat_ids": pstat.index.to_numpy(),
            **{f"pstat_{c}": pstat[c].to_numpy() for c in pstat.columns}})
        np.savez_compressed("embed_model.npz", **{k: (v.astype(str) if v.dtype == object else v) for k, v in Wn.items()})
        print(f"저장: embed_model.npz  타자 {len(bmap)}명, 투수 {len(pmap)}명")
