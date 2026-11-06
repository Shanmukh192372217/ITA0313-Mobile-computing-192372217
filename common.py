import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.grid":True,"grid.alpha":.3})
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"data"); FIGS=os.path.join(HERE,"figs")
def save_json(name,obj):
    with open(os.path.join(DATA,name),"w") as f: json.dump(obj,f,indent=1)
def load_json(name):
    with open(os.path.join(DATA,name)) as f: return json.load(f)
