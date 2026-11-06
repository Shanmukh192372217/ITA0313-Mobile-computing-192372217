import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.linalg import hadamard
from common import *
def box(ax,x,y,w,h,t,fc="#dbe8f7",ec="#2f5d8a",fs=7,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",wrap=True)
def arrow(ax,x1,y1,x2,y2,t=None,c="k",fs=6):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color=c,lw=1))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+.08,t,fontsize=fs,ha="center",color=c)
def blank(w,h):
    f,a=plt.subplots(figsize=(w,h)); a.set_xlim(0,w*2);a.set_ylim(0,h*2);a.axis("off");a.grid(False);return f,a
# ---- Q1 quadrant ----
f,a=plt.subplots(figsize=(6.2,3.8)); a.set_xlim(0,10);a.set_ylim(0,10)
a.axhline(5,c="k",lw=.8);a.axvline(5,c="k",lw=.8)
a.text(.2,9.5,"Wired + portable\n(nomadic)",fontsize=8,va="top",color="#555");a.text(5.2,9.5,"Wireless + mobile\n(true mobility)",fontsize=8,va="top",color="#555")
a.text(.2,.4,"Fixed wired",fontsize=8,color="#555");a.text(5.2,.4,"Fixed wireless",fontsize=8,color="#555")
pts={"(a) cellular tablet":(8,7.3,"#c0504d"),"(b) laptop + Ethernet":(2,7.2,"#4f81bd"),"(c) substation router":(7.8,2.3,"#2e8b57"),"(d) push-to-talk":(8.7,8.9,"#e08a2c"),"(e) offline meter app":(6.2,6.0,"#8064a2"),"(X) my case: parcel tracker":(5.9,5.4,"#000")}
for k,(x,y,c) in pts.items(): a.scatter(x,y,s=60,color=c); a.text(x+.15,y+.1,k,fontsize=7)
a.set_xlabel("wireless access to network  (none → continuous)");a.set_ylabel("device moves while in use (none → continuous)");a.set_xticks([]);a.set_yticks([]);a.grid(False)
plt.tight_layout();plt.savefig(f"{FIGS}/q1_quadrant.png");plt.close()
# ---- Q2 band plan ----
f,a=plt.subplots(2,1,figsize=(7.5,2.6),sharex=True)
for i,(t,n,wd,gl,cl,col) in enumerate((("FDMA: 80 x 25 kHz",80,25,1,2,"#4f81bd"),("TDMA: 10 x 200 kHz carriers x 8 slots",10,200,.5,1,"#e08a2c"))):
    for k in range(n):
        c=col
        if (i==0 and (k in (0,79))) or (i==1 and k in (0,9)): c="#bbbbbb"
        if (i==0 and k in (1,2)) or (i==1 and k==1): c="#7f7f7f"
        a[i].add_patch(Rectangle((k*wd,0),wd*.96,1,fc=c,ec="w"))
    a[i].set_xlim(0,2000);a[i].set_ylim(0,1);a[i].set_yticks([]);a[i].set_title(t+"  (grey light = edge guard, dark grey = control)",fontsize=7,loc="left");a[i].grid(False)
a[1].set_xlabel("frequency within 2 MHz band (kHz)");plt.tight_layout();plt.savefig(f"{FIGS}/q2_band.png");plt.close()
# ---- Q3 block diagram + chip waveform ----
f=plt.figure(figsize=(9,3.3)); a=f.add_axes([0,.45,1,.55]); a.set_xlim(0,18);a.set_ylim(0,3);a.axis("off");a.grid(False)
xs=[.2,2.4,4.6,6.8,10,12.2,14.4]
labels=["data bit b_k\n(±1, Rb)","× code c_k\n(N chips, Rc=N·Rb)","Σ other users\n+ noise n(t)","received\nr = Σ A_j b_j c_j + n","× c_k (local copy)","integrate over\nN chips (÷N)","sign() →\ndecision"]
cols=["#dbe8f7","#dbe8f7","#fde9d9","#fde9d9","#e2f0d9","#e2f0d9","#e2f0d9"]
for x,l,c in zip(xs,labels,cols): box(a,x,1,1.9,1.2,l,fc=c,fs=6)
for i in range(6): arrow(a,xs[i]+1.9,1.6,xs[i+1],1.6)
a.text(.2,2.6,"TRANSMITTER (each handset)",fontsize=7,color="#2f5d8a");a.text(10,2.6,"BASE-STATION CORRELATOR (one per user)",fontsize=7,color="#2e6b2e");a.text(4.6,2.6,"CHANNEL",fontsize=7,color="#a05a00")
a.text(0.2,.35,"Processing gain G = Rc/Rb = N  (10log10 N dB).  Despread output z = b_k·(c_k·c_k)/N + Σ_j≠k A_j b_j ρ_jk + n' ,  ρ_jk = c_j·c_k/N  (=0 for Walsh in sync).",fontsize=6.5)
a2=f.add_axes([.07,.07,.4,.3]); H=hadamard(8); b=np.array([1,-1,1]); c1=H[1]; c2=H[2]
tx1=np.repeat(b,8)*np.tile(c1,3); tx2=np.repeat(np.array([-1,-1,1]),8)*np.tile(c2,3)
a2.step(range(24),tx1+3,where="post",label="user1 chips");a2.step(range(24),tx2,where="post",label="user2 chips");a2.step(range(24),tx1+tx2-3,where="post",label="sum at receiver",color="k")
a2.set_yticks([]);a2.set_xlabel("chip index (N=8, 3 bits)",fontsize=7);a2.legend(fontsize=5,loc="upper right",ncol=3);a2.grid(False)
a3=f.add_axes([.58,.07,.38,.3]); N=16;Hn=hadamard(N)/np.sqrt(N); G=np.abs(Hn@Hn.T);a3.imshow(G,cmap="Blues");a3.set_title("|ρ| Walsh N=16 (identity ⇒ orthogonal)",fontsize=7);a3.grid(False);a3.tick_params(labelsize=6)
plt.savefig(f"{FIGS}/q3_block.png");plt.close()
# ---- Q4 allocation heatmaps ----

# ---- Q4 allocation maps (one realisation, seed 42) ----
U,K=8,64; SNR=np.array([24,21,18,15,12,9,6,3.]); RM=2.0*4
rg=np.random.default_rng(42); L=16; pdp=np.exp(-np.arange(L)/3.0); pdp/=pdp.sum()
hh=(rg.normal(size=(1,U,L))+1j*rg.normal(size=(1,U,L)))/np.sqrt(2)*np.sqrt(pdp)
g=np.abs(np.fft.fft(hh,64,axis=-1))**2*(10**(SNR/10))[None,:,None]; r=np.log2(1+g)[0]
contig=np.repeat(np.arange(U),8); inter=np.arange(K)%U; best=r.argmax(0)
tot=np.zeros(U); free=np.ones(K,bool); mr=np.empty(K,int)
while free.any():
    need=np.where(tot<RM)[0]
    if len(need)==0: break
    u=need[tot[need].argmin()]; k=int(np.where(free,r[u],-1).argmax()); mr[k]=u; free[k]=False; tot[u]+=r[u,k]
for k in np.where(free)[0]: mr[k]=int(r[:,k].argmax())
f,ax=plt.subplots(4,1,figsize=(8,3.6),sharex=True)
for a_,(t,al) in zip(ax,(("Contiguous blocks",contig),("Interleaved RR",inter),("Best-subcarrier",best),("Min-rate + max-sum",mr))):
    a_.imshow(al[None,:],aspect="auto",cmap="tab10",vmin=0,vmax=9,extent=(0,64,0,1));a_.set_yticks([]);a_.set_ylabel(t,rotation=0,ha="right",va="center",fontsize=6);a_.grid(False)
ax[-1].set_xlabel("subcarrier index (colour = user 1..8; user 8 has the weakest mean SNR)");plt.tight_layout();plt.savefig(f"{FIGS}/q4_map.png");plt.close()
# ---- Q7 flowchart ----
f,a=blank(9,4.6)
box(a,.3,6.8,2.6,.9,"Device has data\n(burst starts)",fc="#fff2cc",ec="#b8860b")
box(a,3.6,6.8,2.6,.9,"Holds a channel?\n(T_hold timer running)",fc="#fff2cc",ec="#b8860b")
box(a,7.4,6.8,2.8,.9,"Send burst on held channel\n(0 signalling)",fc="#e2f0d9",ec="#2e6b2e")
box(a,3.6,5.2,2.6,.9,"Send RACH request\n(slotted ALOHA, backoff 1/n)",fc="#dbe8f7")
box(a,3.6,3.6,2.6,.9,"Collision?\nno: BS receives request",fc="#fff2cc",ec="#b8860b")
box(a,3.6,2.0,2.6,.9,"Free channel in pool?",fc="#fff2cc",ec="#b8860b")
box(a,7.4,2.0,2.8,.9,"Reject: burst blocked\n(count blocking)",fc="#f8d7d7",ec="#a33")
box(a,7.4,3.6,2.8,.9,"Collision: retry\nin a later frame",fc="#fde9d9",ec="#a05a00")
box(a,3.6,.4,2.6,.9,"GRANT channel,\nsend burst",fc="#e2f0d9",ec="#2e6b2e")
box(a,.3,.4,2.8,.9,"Burst ends: start T_hold\n(T_hold=0: release msg)",fc="#dbe8f7")
box(a,.3,2.0,2.8,.9,"Timer expires:\nrelease channel to pool",fc="#dbe8f7")
arrow(a,2.9,7.25,3.6,7.25);arrow(a,6.2,7.25,7.4,7.25,"yes");arrow(a,4.9,6.8,4.9,6.1,"no");arrow(a,4.9,5.2,4.9,4.5);arrow(a,4.9,3.6,4.9,2.9,"no collision")
arrow(a,6.2,4.05,7.4,4.05,"collision");arrow(a,8.8,4.5,6.0,5.5,"retry");arrow(a,6.2,2.45,7.4,2.45,"no");arrow(a,4.9,2.0,4.9,1.3,"yes");arrow(a,3.6,.85,3.1,.85);arrow(a,1.7,1.3,1.7,2.0)
a.text(.3,8.2,"Redesigned access: on-demand burst-level channel allocation with hold timer",fontsize=8,fontweight="bold")
plt.savefig(f"{FIGS}/q7_flow.png",bbox_inches="tight");plt.close()
# ---- Q7 message sequence ----
f,a=plt.subplots(figsize=(7,3.2));a.set_xlim(0,10);a.set_ylim(0,8);a.axis("off");a.grid(False)
for x,t in ((1.5,"Device"),(5,"Base station\n(channel pool)"),(8.5,"Other device")):
    a.text(x,7.6,t,ha="center",fontsize=8,fontweight="bold");a.plot([x,x],[0.3,7.2],c="#888",lw=1,ls="--")
def m(y,x1,x2,t,c="k"):
    a.annotate("",xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle="->",color=c));a.text((x1+x2)/2,y+.12,t,ha="center",fontsize=6.5,color=c)
m(6.6,1.5,5,"1. CHANNEL-REQ (RACH slot)");m(5.8,8.5,5,"1'. CHANNEL-REQ: COLLISION","#c0504d")
a.text(7,5.3,"both back off (prob 1/n)",fontsize=6,color="#c0504d");m(4.6,1.5,5,"2. CHANNEL-REQ (retry, success)")
m(3.7,5,1.5,"3. GRANT (channel k)");a.plot([1.1,1.9],[2.9,2.9],c="#2e8b57",lw=5);a.text(2.5,2.8,"burst on ch k",fontsize=6.5)
m(1.9,1.5,5,"4. RELEASE (burst ended, T_hold=0)");a.text(5.3,1.3,"pool back to 76 free",fontsize=6.5)
plt.tight_layout();plt.savefig(f"{FIGS}/q7_msc.png");plt.close()
