import numpy as np

#templates = np.load('/media/mariandawud/Data/results/mds2_res/flows-new/offline_300/templates.npy', allow_pickle=True)
#st = np.load('/data/results/mS51_res/flows_final_shift/batch32768/online_3600/st.npy')
#wpca = np.load('/data/results/mS51_res/flows_final_shift/batch20480/offline_3600/wPCA.npy', allow_pickle=True)
#sim_temp = np.load('/data/results/mP79_res/IIR/time/3600/5peels/2ndOrd-filt/spike_detection_templates.npy')
#data1 = np.load('/data/results/mP79_res/flows_final_shift/updated_dshift/gui-online-shift/export/Wall_nm.npy', allow_pickle=True)
#data2 = np.load('/data/results/mP79_res/flows_final_shift/offline_3600/Wall.npy', allow_pickle=True)
#data3 = np.load('/data/results/mP79_res/mp79-det-3pc-frozen-temp-2peels-7.6/Wall.npy', allow_pickle=True)

#data = np.load("/data/results/mP79_res/flows_c/2peels/V3.5/spike_times.npy")
#temp = np.load("/data/results/mP79_res/flows_c/2peels/V3.5/spike_templates.npy")
#amp = np.load("/data/results/mP79_res/flows_c/2peels/V3.5/amplitudes.npy")
disp = np.load("/data/results/mP79_res/IIR/updated_clu/baseline_det/1/spike_times.npy")
disp1 = np.load("/data/results/mP79_res/IIR/updated_clu/baseline_det/1/spike_clusters.npy")
#cf = fopen("/data/results/mP79_res/flows_c/4peels/cf_trace.txt")
#chan_ind = np.load("/data/results/mds2_res/mds2-det-3pc-frozen-temp-2peels-7.6/pc_feature_ind.npy")
#temp = np.load("/data/results/mP79_res/flows_c/spike_templates.npy")
#th = np.load("/data/results/mP79_res/flows_c/amplitudes.npy")
#wrot = th = np.load("/data/results/mP79_res/flows_final_shift/zero_dshift/offline_3600/export/Wall.npy", allow_pickle=True)*32768
#online = np.load("/data/results/mP79_res/flows_final_shift/updated_dshift/gui-online-shift/FIR/st.npy")
#on_times = np.load("/data/results/mP79_res/flows_final_shift/updated_dshift/gui-online-shift/FIR/spike_times.npy")

# gui1 = np.load("/data/results/mP79_res/IIR/time/3600/1peel/2ndOrd-filt/spike_templates.npy")
# gui_times1 = np.load("/data/results/mP79_res/IIR/time/3600/1peel/2ndOrd-filt/spike_times.npy")

# gui2 = np.load("/data/results/mP79_res/IIR/time/3600/2peels/2ndOrd-filt/spike_templates.npy")
# gui_times2 = np.load("/data/results/mP79_res/IIR/time/3600/2peels/2ndOrd-filt/spike_times.npy")

# gui3 = np.load("/data/results/mP79_res/IIR/time/3600/3peels/2ndOrd-filt/spike_templates.npy")
# gui_times3 = np.load("/data/results/mP79_res/IIR/time/3600/3peels/2ndOrd-filt/spike_times.npy")

# gui4 = np.load("/data/results/mP79_res/IIR/time/3600/4peels/2ndOrd-filt/spike_templates.npy")
# gui_times4 = np.load("/data/results/mP79_res/IIR/time/3600/4peels/2ndOrd-filt/spike_times.npy")

# gui5 = np.load("/data/results/mP79_res/IIR/time/3600/5peels/2ndOrd-filt/spike_templates.npy")
# gui_times5 = np.load("/data/results/mP79_res/IIR/time/3600/5peels/2ndOrd-filt/spike_times.npy")

# gui7 = np.load("/data/results/mP79_res/IIR/time/3600/7peels/2ndOrd-filt/spike_templates.npy")
# gui_times7 = np.load("/data/results/mP79_res/IIR/time/3600/7peels/2ndOrd-filt/spike_times.npy")

# gui10 = np.load("/data/results/mP79_res/IIR/time/3600/10peels/2ndOrd-filt/spike_templates.npy")
# gui_times10 = np.load("/data/results/mP79_res/IIR/time/3600/10peels/2ndOrd-filt/spike_times.npy")

#offline = np.load("/data/results/mS51_res/flows_final_shift/zero_dshift/offline_4800/spike_times.npy")
#ner_chan = np.load("/data/results/mP79_res/mp79-det-3pc-26.6/pc_feature_ind.npy")
#print(np.count_nonzero(cf))
#print(max(st[:,1]))
#print(np.max(data1), np.min(data1))
#print(np.max(data2), np.min(data2))
#print(np.max(data3), np.min(data3))
#print(online.shape)
#print(data[:400])
#print(times[:400])
#print(np.unique(gui5).shape)
#print(amps[:400])
#print(data[:100])
#print(temp[:100])
#print(online[10:100])
#print(on_times[10:100])
#mask = np.where(gui_times2>72000000)
#print(mask)
#print(gui_times[mask])
#print(ner_chan[29])
k=866138
i1=475142
i2=866138
i3=1204650
i4=1496500
i5=1750581
i7=2160211
i10=2599244
#print(gui1[j:j+100])
#print(gui_times1[j:j+100])
print("2p baseline", "\t\t", "c", "\t\t", "1p baseline")
#for i in range(0,100):
#	print(gui2[i2+i], "|", gui_times2[i2+i], "\t\t", data[i], "|", temp[i], "|", round(amp[i],2))
#print(amps[595200:595300])
#print(chan_ind.shape)
#print(chan_ind[80:100])         # first few rows
#print(np.unique(chan_ind, axis=0).shape[0], "unique rows out of", len(chan_ind))
#print(gui.shape)
#print(online.shape)
#print(offline.shape)
#print(data1.min(), np.median(data1), data1.max())
#print(gui[837971:838071])
#print(gui_times[837971:838071])
for i in range(300):
	print(disp[i], "||", disp1[i])
print("\n==========================================\n")
print(f"data: {data.nbytes} bytes.") 
print(f"data: {temp.nbytes} bytes.") 
print(f"amp: {amp.nbytes} bytes.") 