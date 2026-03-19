import subprocess
from pathlib import Path

BIN = "/data/mDS2/dat/mDS2_02/mDS2_02.dat"
PROBE = "/home/mariandawud/kilosort/mds2/probe_mds2.json"

GUI_PHY = "/data/results/mds2_res/mds2-27.2-deterministic"
FLOW_BASE = "/data/results/mds2_res/flows_final_8channels"
COMPARE_BASE = "/data/results/mds2_res/comparison_final_8channels"

FS = 20000
NCHAN = 63
DTYPE = "int16"

# Use these if your folders are named online_1200, online_1800, ...
HOUR_SECS = [300, 600, 1200, 1800, 2400, 3600, 15300, 15600]


def run_cmd(cmd: list[str]):
    """Run a command sequentially, raising if it fails."""
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_flows(hour_sec: int):
    out_offline = f"{FLOW_BASE}/offline_{hour_sec}"
    out_online = f"{FLOW_BASE}/online_{hour_sec}"
    cmd = [
        "python", "flows.py",
        "--bin", BIN,
        "--probe", PROBE,
        "--out_offline", out_offline,
        "--out_online", out_online,
        "--fs", str(FS),
        "--nchan", str(NCHAN),
        "--HOUR_SEC", str(hour_sec),
    ]
    run_cmd(cmd)


def run_compare(t_start: int, tol: int = 0):
    flow_phy = f"{FLOW_BASE}/online_{t_start}"
    out_dir = f"{COMPARE_BASE}/{t_start}-comp"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "compare_templates_timewindow.py",
        "--gui_phy", GUI_PHY,
        "--flow_phy", flow_phy,
        "--tol", str(tol),
        "--out", out_dir,
        "--t_start", str(t_start),
    ]
    run_cmd(cmd)


if __name__ == "__main__":
    for t in HOUR_SECS:
        run_flows(t)

    for t in HOUR_SECS:
        run_compare(t_start=t, tol=12)
