import numpy as np
from phy import IPlugin


class ISIIndexPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """
        Adds cluster metrics:
          - isi_index: refractory metric defined as:
              N(0-2ms) / ( N(0-20ms) * (2/20) )
            where N(a-b) is the count of ISIs in (a, b] ms.
          - isi_pass: 1.0 if isi_index < 0.2 else 0.0

        This matches the paper description:
          "count in the first 2 ms below 0.2 of the expected count
           given the counts in the first 20 ms"
        """

        ISI1_MS = 2.0
        ISI2_MS = 20.0
        THRESH = 0.2

        def isi_index(cluster_id):
            # controller.get_spike_times(cluster_id).data is typically in seconds
            t = controller.get_spike_times(cluster_id).data
            if t is None or len(t) < 3:
                return np.nan

            t = np.asarray(t, dtype=np.float64)
            # make sure sorted (should already be, but safe)
            t.sort()

            isi_ms = np.diff(t) * 1000.0  # convert to ms

            n_0_20 = np.sum((isi_ms > 0) & (isi_ms <= ISI2_MS))
            n_0_2  = np.sum((isi_ms > 0) & (isi_ms <= ISI1_MS))

            expected_0_2 = n_0_20 * (ISI1_MS / ISI2_MS)  # 2/20 = 0.1

            # If there are no ISIs in 0–20 ms, expected is 0 -> metric undefined
            if expected_0_2 <= 0:
                return np.nan

            return float(n_0_2 / expected_0_2)

        def isi_pass(cluster_id):
            v = isi_index(cluster_id)
            if not np.isfinite(v):
                return np.nan
            return float(v < THRESH)

        # Memcache so it persists between sessions (fast)
        controller.cluster_metrics["isi_index"] = controller.context.memcache(isi_index)
        controller.cluster_metrics["isi_pass"] = controller.context.memcache(isi_pass)
