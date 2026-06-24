import numpy as np
"""
script for comparing online flows.py with biquad filter vs kilosort's IIR
Need to run flows.py with BiquadFiltered class.
"""
st_ks = np.load("/data/results/mS51_res/flows_final_shift/updated_dshift/gui-online-shift/FIR/st.npy")
st_bq = np.load("/data/results/mS51_res/flows_final_shift/updated_dshift/gui-online-shift/IIR/st.npy")
a = st_ks[np.argsort(st_ks[:,0])]        # (N,3): time, template, amp
bb = st_bq[np.argsort(st_bq[:,0])]
tb = bb[:,0]; tol = 5                      # samples
same = diff = unmatched = 0
for t, tid in a[:, :2]:
    lo = np.searchsorted(tb, t-tol); hi = np.searchsorted(tb, t+tol)
    if hi > lo:
        j = lo + np.argmin(np.abs(tb[lo:hi]-t))
        same += (bb[j,1] == tid); diff += (bb[j,1] != tid)
    else:
        unmatched += 1
print(f"KS spikes: {len(a)}  biquad spikes: {len(bb)}")
print(f"matched & same template: {same}  matched & different: {diff}")
print(f"identity preserved: {same/(same+diff):.3f}")
print(f"KS spikes with no biquad match: {unmatched}")