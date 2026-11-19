"""사례 9: 고르는 데 쓰지 않은 타자 103명으로 다시 검증 (두 시즌 120스윙 이상, 한 시즌이라도 300 미만)"""
import pickle, numpy as np, pandas as pd
import build_batter_swing as BS, build_swing_hier as SH
SW=BS.SW
C=pickle.load(open("swing_cache.pkl","rb")); s24,s25=C[2024],C[2025]
old=pickle.load(open("batter_val_models.pkl","rb")); new=pickle.load(open("swing_hier_2024.pkl","rb"))
M={"F":old["F"],"E":old["E"],"B":old["B"],"G":new["G"],"H":new["H"]}
a,b=BS.batter_summary(s24),BS.batter_summary(s25)
j=a.join(b,lsuffix="_a",rsuffix="_b",how="inner")
j=j[(j.n_a>=120)&(j.n_b>=120)&((j.n_a<300)|(j.n_b<300))]   # 286명에 안 들어간 타자들 = 고르는 데 안 쓴 검증 집단
print("새 검증 집단", len(j))
for c in SW+["whiff","xwcon","xw_sw"]:
    j["d_"+c]=j[c+"_b"]-j[c+"_a"]; j["d_"+c]-=j["d_"+c].median()
smp=s24[s24.batter.isin(j.index)].groupby("batter",group_keys=False).sample(frac=1,random_state=0).groupby("batter").head(250)
shift=j[["d_"+c for c in SW]].rename(columns=lambda c:c[2:])
x=smp.copy()
for c in SW:
    add=x.batter.map(shift[c]); x[c]+=add; x["m_"+c]+=add
D={}
for k,m in M.items():
    d=(SH.predict(m,x)-SH.predict(m,smp)).groupby(smp.batter).mean(); D[k]=d-d.median()
D["F+E"]=(D["F"]+D["E"])/2
rng=np.random.default_rng(2); rows=[]
for tgt,col in (("whiff","miss"),("xwcon","xwcon"),("xw_sw","xw_sw")):
    act=j["d_"+tgt].to_numpy(); line=tgt
    for k,d in D.items():
        p=d[col].reindex(j.index).to_numpy(); r,ci=BS.corr_ci(p,act); sl=np.polyfit(p,act,1)[0]
        line+=f" | {k} {r:+.3f}[{ci[0]:+.2f},{ci[1]:+.2f}] b{sl:+.2f}"; rows.append({"target":tgt,"model":k,"r":r,"lo":ci[0],"hi":ci[1],"slope":sl,"n":len(j)})
    print(line)
    pf=D["F"][col].reindex(j.index).to_numpy(); pe=D["F+E"][col].reindex(j.index).to_numpy(); ds=[]
    for _ in range(2000):
        i=rng.integers(0,len(act),len(act)); ds.append(np.corrcoef(pe[i],act[i])[0,1]-np.corrcoef(pf[i],act[i])[0,1])
    d0=np.corrcoef(pe,act)[0,1]-np.corrcoef(pf,act)[0,1]; lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"   F+E − F: {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"); rows.append({"target":tgt,"model":"F+E - F","r":d0,"lo":lo,"hi":hi,"n":len(j)})
pd.DataFrame(rows).to_csv("val_swing_hier_holdout.csv",index=False)
