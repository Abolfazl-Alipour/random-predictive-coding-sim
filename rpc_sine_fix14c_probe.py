"""Resume of the fix14c ridge probe (crashed on a proto_n constructor bug,
now fixed in rpc_sine_fix14c.py). Imports probe_delta and runs the trained
vs frozen conditions only, 3 seeds each.
"""
import numpy as np
from rpc_sine_fix14c import probe_delta

if __name__ == "__main__":
    for cond, plast in [("trained", True), ("frozen", False)]:
        ds = [probe_delta(s, plast) for s in range(3)]
        print(f"probe delta256 d1 next-token {cond}: mean={np.mean(ds):.3f} "
              f"range=[{min(ds):.3f},{max(ds):.3f}] "
              f"accs={[round(d, 3) for d in ds]}", flush=True)
