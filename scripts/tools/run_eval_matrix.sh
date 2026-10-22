#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the SimFoundry policy evaluation under several named conditions, one after another, so
# the conditions can be compared (see summarize_eval.py). Each condition gets its own results
# directory because its name is passed as s15_eval.checkpoint.
#
# Needs: the simfoundry conda env active, and a GR00T policy server already listening on :5555
# (started with --use-sim-policy-wrapper).
#
# Usage:
#   scripts/tools/run_eval_matrix.sh <n_episodes> <timeout_s> <condition> [<condition> ...]
#   scripts/tools/run_eval_matrix.sh 5 30 base floor prompt_short horizon4 grip_noinvert
#
# Env overrides: SCENE_JSON, SCENE_NAME, ROOT_DIR, PORT, LOG_DIR, TASK, EVAL_PROMPT.
# TASK and EVAL_PROMPT let the same runner drive one of the published example scenes, e.g. as a
# positive control for the policy wiring:
#   TASK=droid/droid_desk_put_marker_in_cup SCENE_NAME=droid_desk_put_marker_in_cup \
#   SCENE_JSON=<abs path to the example scene state json> EVAL_PROMPT="put the marker in the cup" \
#   scripts/tools/run_eval_matrix.sh 3 30 base
set -u

if [[ $# -lt 3 ]]; then
  sed -n '3,16p' "$0"
  exit 2
fi
N_EPISODES="$1"
TIMEOUT_S="$2"
shift 2

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE_NAME="${SCENE_NAME:-put_cup_test}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/Data}"
PORT="${PORT:-5555}"
SCENE_JSON="${SCENE_JSON:-${ROOT_DIR}/${SCENE_NAME}/s14_og/reconstructed_og_scene.json}"
LOG_DIR="${LOG_DIR:-/data/projects/simfoundry/eval_matrix}"
TASK="${TASK:-droid/droid_desk_put_cup_in_bowl}"
DEFAULT_PROMPT="${EVAL_PROMPT:-put the white cup in the orange bowl}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

for cond in "$@"; do
  PROMPT="${DEFAULT_PROMPT}"
  EXTRA=()
  case "${cond}" in
    base) ;;
    floor) EXTRA=("s15_eval.floor_plane_visible=true") ;;
    prompt_short) PROMPT="put the cup in the bowl" ;;
    horizon4) EXTRA=("s15_eval.execute_horizon=4") ;;
    # gripper_invert is read with cfg.s15_eval.get("gripper_invert", True), so it is not in
    # real2sim_cfg.yaml and has to be appended with '+'.
    grip_noinvert) EXTRA=("+s15_eval.gripper_invert=false") ;;
    # Skybox brightness sweep: scales the scene's default dome-light intensity.
    sky_half) EXTRA=("+s15_eval.skybox_intensity_scale=0.5") ;;
    sky_quarter) EXTRA=("+s15_eval.skybox_intensity_scale=0.25") ;;
    *) echo "[matrix] unknown condition '${cond}', skipping" >&2; continue ;;
  esac

  log="${LOG_DIR}/${cond}.log"
  echo "[matrix] $(date +%H:%M:%S) ${cond}: ${N_EPISODES} episodes, ${TIMEOUT_S}s, prompt='${PROMPT}' ${EXTRA[*]:-} -> ${log}"
  python scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py \
    task="${TASK}" \
    scene_name="${SCENE_NAME}" root_dir="${ROOT_DIR}" \
    s15_eval.scene_json="${SCENE_JSON}" \
    "s15_eval.prompt=\"${PROMPT}\"" \
    s15_eval.policy=gr00t s15_eval.port="${PORT}" \
    s15_eval.external_sensors_cfg=nv_franka_droid \
    '+s15_eval.video_resolution=[360,640]' \
    s15_eval.n_episodes="${N_EPISODES}" s15_eval.timeout_s="${TIMEOUT_S}" \
    s15_eval.save_video=true \
    s15_eval.checkpoint="${cond}" \
    "${EXTRA[@]}" \
    > "${log}" 2>&1
  status=$?
  echo "[matrix] $(date +%H:%M:%S) ${cond}: exit ${status}  $(tr '\r' '\n' < "${log}" | grep -m1 'Success rate' || echo 'no success-rate line')"
done
echo "[matrix] done"
