"""Q5 - SDMA. ULA, d=lambda/2, steered to user A (theta_A). Array factor:
 AF(th)=sum_{n=0}^{N-1} exp(j*n*psi), psi=pi*(sin th - sin thA) ; |AF|/N = |sin(N psi/2)/(N sin(psi/2))|.
PREDICTIONS: null-null BW ~ 2*asin(2/N) at broadside (HPBW~101.5deg/N), first sidelobe -> -13.3 dB (uniform taper) independent of N;
 min separation for SIR>=10 dB shrinks ~1/N (N=16 ~ a few deg), while 120-deg sector and omni cannot separate users inside the same sector (0 dB). 
 Steering off broadside widens the beam by 1/cos(thA)."""
import numpy as np, pandas as pd
from common import *
def af(N,th,thA):
    psi=np.pi*(np.sin(np.radians(th))-np.sin(np.radians(thA)))
    num=np.sin(N*psi/2); den=N*np.sin(psi/2)
    out=np.where(np.abs(den)<1e-12,1.0,num/np.where(np.abs(den)<1e-12,1,den)); return np.abs(out)
def sector_att_db(delta,th3=65,am=20): return np.minimum(12*(delta/th3)**2,am)
Ns=(2,4,8,16); th=np.linspace(-90,90,18001)
rows=[]
for N in Ns:
    a=af(N,th,0); db=20*np.log10(np.maximum(a,1e-6))
    # HPBW
    inside=th[a>=1/np.sqrt(2)]; hp=inside.max()-inside.min()
    # first null
    fn=np.degrees(np.arcsin(2/N)); 
    # sidelobe level: highest local max outside first null
    pk=[i for i in range(1,len(a)-1) if a[i]>a[i-1] and a[i]>=a[i+1] and abs(th[i])>fn+0.2]
    sll=max(db[i] for i in pk) if pk else np.nan
    rows.append(dict(N=N,HPBW_deg=hp,HPBW_theory=101.5/N,null_to_null_deg=2*fn,SLL_dB=sll,peak_gain_dBi_rel=10*np.log10(N)))
bw=pd.DataFrame(rows); print(bw.round(2)); bw.to_csv(f"{DATA}/q5_beam.csv",index=False)
seps=np.arange(5,61,5.)
res=[]
for N in Ns:
    for s in seps:
        sir=-20*np.log10(max(af(N,np.array([s]),0)[0],1e-6)); res.append(dict(system=f"ULA N={N}",sep=s,SIR_dB=sir))
for s in seps:
    res.append(dict(system="Omni",sep=s,SIR_dB=0.0)); res.append(dict(system="120deg sector",sep=s,SIR_dB=float(sector_att_db(s))))
sr=pd.DataFrame(res); sr.to_csv(f"{DATA}/q5_sir.csv",index=False)
print(sr.pivot(index="sep",columns="system",values="SIR_dB").round(1).to_string())
# min separation: smallest s0 such that SIR>=thr for ALL s>=s0 (up to 60deg), fine grid
fine=np.arange(0.1,60.01,0.1); thr_list=(10,15)
ms=[]
for N in Ns:
    sir=-20*np.log10(np.maximum(af(N,fine,0),1e-6))
    for thr in thr_list:
        ok=sir>=thr; 
        # first sep where first-crossing; and sep after which always ok
        first=fine[np.argmax(ok)] if ok.any() else np.nan
        bad=np.where(~ok)[0]; always=fine[bad.max()+1] if len(bad) and bad.max()+1<len(fine) else (np.nan if len(bad) else fine[0])
        ms.append(dict(N=N,thr_dB=thr,first_cross_deg=first,always_ok_beyond_deg=always))
msd=pd.DataFrame(ms); print(msd.round(1)); msd.to_csv(f"{DATA}/q5_minsep.csv",index=False)
# steering off broadside
off=[]
for N in Ns:
    for tA in (0,30,60):
        a=af(N,th,tA); inside=th[a>=1/np.sqrt(2)]; off.append(dict(N=N,steer_deg=tA,HPBW_deg=inside.max()-inside.min()))
offd=pd.DataFrame(off); print(offd.pivot(index="N",columns="steer_deg",values="HPBW_deg").round(1)); offd.to_csv(f"{DATA}/q5_steer.csv",index=False)
# angular spread Monte Carlo (each user's true bearing = nominal + N(0,sigma))
rng=np.random.default_rng(2024); M=20000; mc=[]
for N in (8,16):
    for sig in (0,2,5):
        for s in (3,6,10,15,20,30):
            ta=rng.normal(0,sig,M); tb=s+rng.normal(0,sig,M)
            # beam steered to *estimated* bearing of A (nominal 0); B actual at tb relative to beam
            sir=-20*np.log10(np.maximum(af(N,tb,0),1e-6)) + 20*np.log10(np.maximum(af(N,ta,0),1e-6))  # signal loss for A off-beam too
            mc.append(dict(N=N,sigma=sig,sep=s,P_SIR_ge10=float((sir>=10).mean()),median_SIR=float(np.median(sir))))
mcd=pd.DataFrame(mc); mcd.to_csv(f"{DATA}/q5_spread.csv",index=False); print(mcd[mcd.N==16].pivot(index="sep",columns="sigma",values="P_SIR_ge10").round(2))
save_json("q5.json",dict(beam=rows,minsep=ms,steer=off))
# figures
fig=plt.figure(figsize=(9.5,3.3))
ax=fig.add_subplot(1,3,1,projection="polar"); 
for N in Ns:
    t=np.radians(np.linspace(-90,90,2000)); a=np.maximum(20*np.log10(np.maximum(af(N,np.degrees(t),0),1e-6)),-40)+40
    ax.plot(t,a,label=f"N={N}")
ax.set_theta_zero_location("N");ax.set_thetamin(-90);ax.set_thetamax(90);ax.set_rticks([10,20,30,40]);ax.set_yticklabels(["-30","-20","-10","0 dB"],fontsize=6);ax.legend(fontsize=6,loc="lower left",bbox_to_anchor=(-.2,-.05));ax.set_title("Array patterns (steered 0°)",fontsize=8)
ax2=fig.add_subplot(1,3,2)
for N in Ns: ax2.plot(th,np.maximum(20*np.log10(np.maximum(af(N,th,0),1e-6)),-45),label=f"N={N}")
ax2.set_xlim(-60,60);ax2.set_ylim(-45,3);ax2.set_xlabel("angle (deg)");ax2.set_ylabel("gain rel. peak (dB)");ax2.axhline(-10,ls="--",c="k",lw=.8);ax2.legend(fontsize=6)
ax3=fig.add_subplot(1,3,3)
for sy in ("Omni","120deg sector","ULA N=2","ULA N=4","ULA N=8","ULA N=16"):
    d=sr[sr.system==sy]; ax3.plot(d.sep,d.SIR_dB,marker="o",ms=3,label=sy)
ax3.axhline(10,ls="--",c="k",lw=.8);ax3.set_xlabel("angular separation A–B (deg)");ax3.set_ylabel("SIR (dB)");ax3.legend(fontsize=6)
plt.tight_layout();plt.savefig(f"{FIGS}/q5_patterns.png")
