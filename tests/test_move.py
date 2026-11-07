import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

import pickle, sys
import pandas as pd
import mechanics as MC
S = pickle.load(open("swing_model.pkl", "rb"))
samp = pickle.load(open("pitch_samples.pkl", "rb"))
rep = pickle.load(open("pitcher_report.pkl", "rb"))
phys = pickle.load(open("physics.pkl", "rb"))
move = MC.load_move()
print("계수 (ivb, rel_z/side/ext):", {k: v.round(2).tolist() for k, v in move.items() if k[1] == "ivb"})
print("계수 (hb):", {k: v.round(2).tolist() for k, v in move.items() if k[1] == "hb"})
print("계수 (velo):", {k: v.round(2).tolist() for k, v in move.items() if k[1] == "velo"})
pd.set_option("display.width", 220)
for nm in sys.argv[1:] or ["Skenes, Paul", "Hader, Josh", "Webb, Logan"]:
    r = rep[rep["player_name"] == nm]
    pid, hand = r["pitcher"].iloc[0], r["p_throws"].iloc[0]
    ph = phys[phys["pitcher"] == pid].set_index("pitch_type")
    best = r[r["is_best"]]["pitch_type"].iloc[0]
    for lab, mv in (("무브 고정", None), ("무브 반영", move)):
        res, types, usage = MC.simulate(S, samp, ph, rep, pid, hand, move=mv)
        print(f"\n=== {nm} [{lab}] best={best}")
        for focus in (best, "all"):
            bp = MC.best_plan(res, focus, types).head(2)
            print(f"  focus {focus}:", bp[["dh", "ds", "de", "d_" + best, "d_all"]].round(2).values.tolist())
        o = res[(res.ds == 0) & (res.de == 0)].set_index("dh")
        cols = ["d_" + t for t in types]
        print("  타점만:", o[cols].round(2).to_string().replace("\n", "\n   "))
        if mv is not None:
            print("  타점+0.5 무브변화:", {t: (round(o.loc[0.5, "ivb_" + t], 2), round(o.loc[0.5, "hb_" + t], 2), round(o.loc[0.5, "velo_" + t], 2)) for t in types})
