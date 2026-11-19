"""사례 9: 지금 모델(F)과 사례 2의 큰 잎 GBM(E)을 평균한 앙상블 — 286명 검증 + '지금 그대로가 최적' 비율 (build_batter_swing.py val, build_swing_hier.py fit 먼저)"""
import pickle, numpy as np, pandas as pd
import build_batter_swing as BS, build_swing_hier as SH
SW=BS.SW
C=pickle.load(open("swing_cache.pkl","rb")); s24,s25=C[2024],C[2025]
old=pickle.load(open("batter_val_models.pkl","rb")); new=pickle.load(open("swing_hier_2024.pkl","rb"))
M={"F":old["F"],"E":old["E"],"B":old["B"],"H":new["H"]}
a,b=BS.batter_summary(s24),BS.batter_summary(s25)
j=a.join(b,lsuffix="_a",rsuffix="_b",how="inner"); j=j[(j.n_a>=300)&(j.n_b>=300)]
for c in SW+["whiff","xwcon","xw_sw"]:
    j["d_"+c]=j[c+"_b"]-j[c+"_a"]; j["d_"+c]-=j["d_"+c].median()
smp=s24[s24.batter.isin(j.index)].groupby("batter",group_keys=False).sample(frac=1,random_state=0).groupby("batter").head(250)
shift=j[["d_"+c for c in SW]].rename(columns=lambda c:c[2:])
x=smp.copy()
for c in SW:
    add=x.batter.map(shift[c]); x[c]+=add; x["m_"+c]+=add
base={k:SH.predict(m,smp) for k,m in M.items()}
D={}
for k,m in M.items():
    d=(SH.predict(m,x)-base[k]).groupby(smp.batter).mean(); D[k]=d-d.median()
D["F+E"]=(D["F"]+D["E"])/2; D["H+E"]=(D["H"]+D["E"])/2
rng=np.random.default_rng(1)
for tgt,col in (("whiff","miss"),("xwcon","xwcon"),("xw_sw","xw_sw")):
    act=j["d_"+tgt].to_numpy(); line=tgt
    for k,d in D.items():
        p=d[col].reindex(j.index).to_numpy(); r,ci=BS.corr_ci(p,act); sl=np.polyfit(p,act,1)[0]
        line+=f" | {k} {r:+.3f}[{ci[0]:+.2f},{ci[1]:+.2f}] b{sl:+.2f}"
    print(line)
    pf=D["F"][col].reindex(j.index).to_numpy(); pe=D["F+E"][col].reindex(j.index).to_numpy()
    ds=[]
    for _ in range(2000):
        i=rng.integers(0,len(act),len(act)); ds.append(np.corrcoef(pe[i],act[i])[0,1]-np.corrcoef(pf[i],act[i])[0,1])
    print(f"   F+E − F: {np.corrcoef(pe,act)[0,1]-np.corrcoef(pf,act)[0,1]:+.3f} [{np.percentile(ds,2.5):+.3f}, {np.percentile(ds,97.5):+.3f}]")
# '지금 그대로가 최적' 비율: 각 스윙 요소 ±0.5 SD (타자 간 표준편차) 12가지 + 그대로
sd=a.loc[a.n>=300,SW].std()
for k in ["B","E","F","H"]:
    best_stay=0; tot=0
    for bt,g in smp.groupby("batter"):
        g=g.head(120); b0=SH.predict(M[k],g)["xw_sw"].mean(); better=False
        for c in SW:
            for sgn in (1,-1):
                h=g.copy(); h[c]+=sgn*0.5*sd[c]; h["m_"+c]+=sgn*0.5*sd[c]
                if SH.predict(M[k],h)["xw_sw"].mean()>b0+1e-4: better=True; break
            if better: break
        tot+=1; best_stay+= (not better)
    print(f"{k}: 지금 그대로가 최적 {best_stay}/{tot}")
