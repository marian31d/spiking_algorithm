from pathlib import Path
import numpy as np
from phy import IPlugin


class PeakToTroughPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """
        Adds a cluster metric: p2t_uV (peak-to-trough amplitude in microvolts)

        It expects a TSV file that contains:
            cluster_id    peak_to_trough_uV

        Default filenames it will try (in the dataset folder):
          - cluster_peak_to_trough_from_raw_uV.tsv
          - cluster_peak_to_trough_uV.tsv
        """

        # -------- find dataset (kilosort output) directory --------
        # Phy/phy2 controller implementations differ slightly; try a few.
        base_dir = None
        for attr in ("dir_path", "data_path"):
            if hasattr(controller, attr):
                try:
                    base_dir = Path(getattr(controller, attr))
                    break
                except Exception:
                    pass

        if base_dir is None:
            # common in phy2 TemplateController
            try:
                base_dir = Path(controller.model.dir_path)
            except Exception:
                base_dir = None

        if base_dir is None:
            # give up gracefully
            print("[PeakToTroughPlugin] Could not determine dataset directory. Metric not loaded.")
            return

        # -------- locate TSV --------
        candidates = [
            base_dir / "cluster_peak_to_trough_from_raw_uV.tsv",
            base_dir / "cluster_peak_to_trough_uV.tsv",
        ]
        tsv_path = next((p for p in candidates if p.exists()), None)
        if tsv_path is None:
            print(f"[PeakToTroughPlugin] No TSV found in {base_dir}. Expected one of:")
            for p in candidates:
                print("  -", p.name)
            print("Metric not loaded.")
            return

        # -------- load TSV into dict: cluster_id -> p2t_uV --------
        # Robust TSV reader without pandas.
        p2t_map = {}
        with tsv_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")

            # accept either:
            #   peak_to_trough_uV
            # or:
            #   peak_to_trough_from_raw_uV / median_p2t_uV etc.
            # We'll prioritize "peak_to_trough_uV" then "median_p2t_uV".
            def col_index(name):
                return header.index(name) if name in header else None

            i_cluster = col_index("cluster_id")
            i_p2t = col_index("peak_to_trough_uV")
            if i_p2t is None:
                i_p2t = col_index("median_p2t_uV")  # from the raw-snippet script
            if i_cluster is None or i_p2t is None:
                print("[PeakToTroughPlugin] TSV missing required columns.")
                print("Header columns:", header)
                print("Need: cluster_id and (peak_to_trough_uV or median_p2t_uV)")
                return

            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                try:
                    cid = int(float(parts[i_cluster]))
                    val = float(parts[i_p2t])
                    p2t_map[cid] = val
                except Exception:
                    continue

        print(f"[PeakToTroughPlugin] Loaded {len(p2t_map)} cluster p2t values from {tsv_path.name}")

        # -------- metric function --------
        def p2t_uV(cluster_id):
            # Return NaN if missing so Phy can display empty/blank
            return float(p2t_map.get(int(cluster_id), np.nan))

        # Memcache so Phy stores it and you don't recompute/load every time
        controller.cluster_metrics["p2t_uV"] = controller.context.memcache(p2t_uV)

