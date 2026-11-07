import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 모델 파일(.pkl)을 저장소 루트에서 읽음

import pickle, sys, time
import pandas as pd
import mechanics as MC

S = pickle.load(open("swing_model.pkl", "rb"))
samp = pickle.load(open("pitch_samples.pkl", "rb"))
rep = pickle.load(open("pitcher_report.pkl", "rb"))
phys = pickle.load(open("physics.pkl", "rb"))
names = sys.argv[1:] or ["Skenes, Paul", "Hader, Josh", "Sale, Chris", "Webb, Logan"]
pd.set_option("display.width", 200)
for nm in names:
    r = rep[rep["player_name"] == nm]
    if r.empty:
        print("없음", nm); continue
    pid, hand = r["pitcher"].iloc[0], r["p_throws"].iloc[0]
    ph = phys[phys["pitcher"] == pid].set_index("pitch_type")
    t0 = time.time()
    res, types, usage = MC.simulate(S, samp, ph, rep, pid, hand)
    best = r[r["is_best"]]["pitch_type"].tolist()
    print(f"\n=== {nm} ({hand}) {time.time()-t0:.1f}s  best={best}  feasible {res['feasible'].sum()}/{len(res)}")
    print(r[["pitch_type","n","usage","velo","rel_z","rel_side","ext","vaa_rel","swstr","rv100_pct"]].round(2).to_string(index=False))
    for focus in (best[:1] + ["all"]):
        bp = MC.best_plan(res, focus, types)
        cols = ["dh","ds","de","score"] + ["d_"+t for t in types] + ["d_all"]
        print(f"-- focus {focus}")
        print(bp[cols].head(4).round(2).to_string(index=False))
    # 개별 축 효과 (다른 건 0)
    for ax in ("dh","ds","de"):
        o = [a for a in ("dh","ds","de") if a != ax]
        s = res[(res[o[0]]==0)&(res[o[1]]==0)].set_index(ax)[["d_"+t for t in types]+["d_all"]]
        print(f"   축 {ax}:\n" + s.round(2).to_string())
    print("   제외 이유 예:", res.loc[~res["feasible"], "why"].value_counts().head(4).to_dict())
