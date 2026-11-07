"""
선수 임베딩 활용 (build_embed.py가 만든 embed_model.npz — numpy만 있으면 됨)
  - 비슷한 선수 찾기 (코사인 유사도)
  - 선수 지도 (주성분 2개로 압축) + 축 해석 (어떤 기본 지표와 관련 있는지)
"""
import os

import numpy as np
import pandas as pd

STAT_KO = {"whiff": "헛스윙%", "chase": "유인구 스윙%", "zswing": "존 안 스윙%", "swing": "스윙%", "rv100": "100구당 득점",
           "ev": "타구 속도", "velo": "직구 구속", "fb_share": "직구 비율"}


def load(path="embed_model.npz"):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    W = {k: z[k] for k in z.files}
    out = {"W": W}
    for kind, pre, emb in (("batter", "bstat", "eb"), ("pitcher", "pstat", "ep")):
        ids, idx = W[f"{kind[0]}_ids"], W[f"{kind[0]}_idx"]
        cols = [str(c) for c in W[f"{pre}_cols"]]
        st = pd.DataFrame({c: W[f"{pre}_{c}"] for c in cols}, index=W[f"{pre}_ids"])
        st = st.reindex(ids)
        for c in st.columns:
            if c not in ("name", "hand", "stand"):
                st[c] = pd.to_numeric(st[c], errors="coerce")
        out[kind] = {"ids": ids, "E": W[emb][idx], "stat": st}
    return out


def _subset(T, kind, min_n):
    t = T[kind]
    keep = (t["stat"]["n"].fillna(0).to_numpy() >= min_n)
    return t["ids"][keep], t["E"][keep], t["stat"][keep]


def similar(T, kind, pid, k=10, min_n=500):
    ids, E, st = _subset(T, kind, min_n)
    if pid not in set(ids):
        ids, E, st = _subset(T, kind, 0)
        if pid not in set(ids):
            return None
    En = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)
    i = int(np.where(ids == pid)[0][0])
    s = En @ En[i]
    o = [j for j in np.argsort(-s) if j != i][:k]
    return pd.DataFrame({"id": ids[o], "sim": s[o]}).join(st.iloc[o].reset_index(drop=True))


def pca_map(T, kind, min_n=1000):
    ids, E, st = _subset(T, kind, min_n)
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    pc = Ec @ Vt[:2].T
    df = pd.DataFrame({"id": ids, "pc1": pc[:, 0], "pc2": pc[:, 1]}).join(st.reset_index(drop=True))
    var = S[:2] ** 2 / (S ** 2).sum()
    axes = []
    for k in ("pc1", "pc2"):
        cors = {c: df[[k, c]].corr().iloc[0, 1] for c in st.columns if c in STAT_KO and df[c].notna().sum() > 20}
        top = sorted(cors.items(), key=lambda x: -abs(x[1]))[:2]
        axes.append(top)
    # 보기 좋게: 첫 축의 가장 강한 지표가 오른쪽으로 갈수록 커지게 부호 맞추기
    for j, k in enumerate(("pc1", "pc2")):
        if axes[j] and axes[j][0][1] < 0:
            df[k] = -df[k]
            axes[j] = [(c, -v) for c, v in axes[j]]
    return df, var, axes


def axis_label(top):
    if not top:
        return ""
    return " · ".join(f"{STAT_KO.get(c, c)} {'↑' if v > 0 else '↓'} (상관 {v:+.2f})" for c, v in top)
