import numpy as np
from pathlib import Path

rd = Path("/media/mariandawud/int_ssd/data/kilosort4/ms51-15.12-new")  # change

ops = np.load(rd/"ops.npy", allow_pickle=True).item()
print("ops scale:", ops.get("scale", None), " ops shift:", ops.get("shift", None))

tpl = np.load(rd/"templates.npy")
print("templates.npy range:", float(tpl.min()), float(tpl.max()))

tu = rd/"templates_unw.npy"
print("templates_unw.npy exists?", tu.exists())
if tu.exists():
    tplu = np.load(tu)
    print("templates_unw.npy range:", float(tplu.min()), float(tplu.max()))
