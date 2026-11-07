import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

import pickle, sys
import pandas as pd
import mechanics as MC
S = pickle.load(open("swing_model.pkl", "rb")); samp = pickle.load(open("pitch_samples.pkl", "rb"))
rep = pickle.load(open("pitcher_report.pkl", "rb")); phys = pickle.load(open("physics.pkl", "rb"))
move = MC.load_move()
pd.set_option("display.width", 200)
for nm in sys.argv[1:] or ["Skenes, Paul", "Hader, Josh"]:
    r = rep[rep["player_name"] == nm]; pid, hand = r["pitcher"].iloc[0], r["p_throws"].iloc[0]
    ph = phys[phys["pitcher"] == pid].set_index("pitch_type")
    res, types = MC.shape_simulate(S, samp, ph, rep, pid, hand)
    print(f"\n=== {nm}  feasible {res['feasible'].mean():.0%}")
    for t in types:
        b = MC.best_shape(res, t).head(1)
        v = res[(res.pitch_type == t) & (res.d_arm == 0) & (res.d_velo == 0)].set_index("d_ivb")["d"].round(2).to_dict()
        vv = res[(res.pitch_type == t) & (res.d_arm == 0) & (res.d_ivb == 0)].set_index("d_velo")["d"].round(2).to_dict()
        print(t, b[["d_ivb", "d_arm", "d_velo", "d", "score"]].round(2).values.tolist(), "ivb축", v, "구속축", vv)
    rr, types, usage = MC.simulate(S, samp, ph, rep, pid, hand, move=move)
    best = r[r.is_best].pitch_type.iloc[0]
    print("릴리스(체인)", best, MC.best_plan(rr, best, types).head(3)[["dh", "ds", "de", "d_" + best, "d_all", "score"]].round(2).values.tolist())
    print("릴리스(체인) all", MC.best_plan(rr, "all", types).head(3)[["dh", "ds", "de", "d_all", "score"]].round(2).values.tolist())
