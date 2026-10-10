#!/usr/bin/env python3
"""
Ask a running GR00T policy server for one action chunk, using dummy observations, without
starting the simulator. Use it to check that the client request format matches the server
(start the server with `--use-sim-policy-wrapper` for the public N1.7 server).

Run inside the `simfoundry` env:
    python scripts/tools/test_gr00t_client.py [--port 5555] [--prompt "put the cup in the bowl"]
"""
import argparse
import time

import numpy as np

from simfoundry.policies.gr00t import Gr00tClient

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=5555)
ap.add_argument("--prompt", default="put the white cup in the orange bowl")
args = ap.parse_args()

client = Gr00tClient(host=args.host, port=args.port, open_loop_horizon=8)
print("policy_mode:", client.policy_mode)

rng = np.random.default_rng(0)
img = lambda: rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
obs = {
    "exterior_image_1_left": img(),
    "exterior_image_2_left": img(),
    "wrist_image_left": img(),
    # Franka "home-ish" pose, the same one the eval logs on its first step.
    "joint_position": np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75], dtype=np.float32),
    "gripper_position": np.array([0.0], dtype=np.float32),
    "eef_9d": np.array([0.4, 0.0, 0.3, 1, 0, 0, 0, 1, 0], dtype=np.float32),
}

for i in range(2):  # the second call shows the steady-state latency
    t0 = time.time()
    out = client.infer(obs, args.prompt)
    dt = time.time() - t0
    a = out["action"]
    print(f"call {i + 1}: {dt:.2f}s  action chunk shape = {a.shape}  dtype = {a.dtype}")
print("first action row (7 joints + gripper):", np.round(out["action"][0], 3))
print("viz image shape:", out["viz"].shape)
print("OK: request/response format matches the server")
