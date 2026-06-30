# spiking_algorithm
This repository includes:
1. Information files for 3 neuronal datasets, mS51, mP79 and mDS2 from datadryad.org database.
2. run files for p;oting, comparisng and running functions on those datasets.
3. A proposed bare-metal C code for implementing the template matching function from Kilosort on a RISC-V processor.

#filter 
It contains matlab code for comparing different filters, IIR and FIR and their effect on data.
and the plots results for all filters.

#kilsoort 
kilosort's original code was edited in a couple places.
1. for determinism - spike_detect.py
2. for filter change - io.py
3. for whitening range (discluding batches) - preprocessing.py
4. for frozen template learning process - run_kilosort.py

#ms51, mp79, mds2
includes probe files, C code for template matching and all exported files from kilosort for its run.

#new-plugins
plugins for PHY rpogram to include additional cluster filtering metrics

#run-files
All sorts of helper functions for all sorts of things such as - plotting, files sizes checks and scrips to display numpy files contents.
Comparison scripts and a subenvironment similar to kilosort but partitioned into two phases, offline and online (flows.py).
