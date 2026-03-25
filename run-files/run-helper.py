import subprocess
from pathlib import Path

class data:
    def __init__(self, BIN, PROBE, NCHAN, GUI_PHY, FLOW_BASE, COMPARE_BASE, HOUR_SECS):
        self.BIN = BIN
        self.PROBE=PROBE
        self.NCHAN=NCHAN
        self.GUI_PHY=GUI_PHY
        self.FLOW_BASE=FLOW_BASE
        self.COMPARE_BASE=COMPARE_BASE
        self.HOUR_SECS=HOUR_SECS

FS = 20000
DTYPE = "int16"



def run_cmd(cmd: list[str]):
    """Run a command sequentially, raising if it fails."""
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_flows(hour_sec: int, data: str):
    out_offline = f"{data.FLOW_BASE}/offline_{hour_sec}"
    out_online = f"{data.FLOW_BASE}/online_{hour_sec}"
    cmd = [
        "python", "flows.py",
        "--bin", data.BIN,
        "--probe", data.PROBE,
        "--out_offline", out_offline,
        "--out_online", out_online,
        "--fs", str(FS),
        "--nchan", str(data.NCHAN),
        "--HOUR_SEC", str(hour_sec),
    ]
    run_cmd(cmd)


def run_compare(t_start: int, data: str, tol: int = 0):
    flow_phy = f"{data.FLOW_BASE}/online_{t_start}"
    out_dir = f"{data.COMPARE_BASE}/{t_start}-comp"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "compare_templates_timewindow_old.py",
        "--gui_phy", data.GUI_PHY,
        "--flow_phy", flow_phy,
        "--tol", str(tol),
        "--out", out_dir,
        "--t_start", str(t_start),
    ]
    run_cmd(cmd)


if __name__ == "__main__":
    ms51 = data("/data/mS51/dat/mS51_49/mS51_49.dat", \
        "/home/mariandawud/kilosort/ms51/probe_ms51_new.json", 135, \
        "/data/results/mS51_res/ms51-deterministic-7.3", \
        "/data/results/mS51_res/flows_final_shift", \
        "/data/results/mS51_res/comparison_final_shift", \
        [300, 600, 1200, 1800, 2400, 3600, 20400, 20700])

    mp79 = data("/data/mP79/dat/mP79_17/mP79_17.dat", \
        "/home/mariandawud/kilosort/mp79/probe_mp79.json", 128, \
        "/data/results/mP79_res/mp79-7.3-deteministic", \
        "/data/results/mP79_res/flows_final_shift", \
        "/data/results/mP79_res/comparison_final_shift", \
        [300, 600, 1200, 1800, 2400, 3600, 13428, 13600])

    mds2 = data("/data/mDS2/dat/mDS2_02/mDS2_02.dat", \
        "/home/mariandawud/kilosort/mds2/probe_mds2.json", 63, \
        "/data/results/mds2_res/mds2-27.2-deterministic", \
        "/data/results/mds2_res/flows_final_edit", \
        "/data/results/mds2_res/comparison_final_edit", \
        [300, 600, 1200, 1800, 2400, 3600, 15300, 15600])

    data_l =[ms51, mp79, mds2]

    for d in data_l:
        for t in d.HOUR_SECS:
            run_flows(hour_sec=t,data=d)

        for t in d.HOUR_SECS:
            run_compare(t_start=t, data=d,tol=12)

    #d=mds2
    #for t in d.HOUR_SECS[6:]:
    #    run_flows(hour_sec=t,data=d)

    #for t in d.HOUR_SECS[:6]:
    #    run_compare(t_start=t, data=d,tol=12)
