#!/usr/bin/env python3
"""
测试千问后端：从示例视频取一帧（或直接给图片），让模型检测物体并输出框，再把框画到图上，
用来确认坐标格式（Gemini 用 [ymin, xmin, ymax, xmax]，0-1000 归一化）。
用法：
  cd /data/SimFoundry && conda activate simfoundry
  python scripts/tools/test_qwen_vlm.py docs/assets/example_videos/PutCupInBowl.mp4
输出：/data/projects/simfoundry/qwen_test/frame.png、boxes.png、reply.txt
"""
import json, os, re, sys
from pathlib import Path

os.environ.setdefault("SIMFOUNDRY_VLM_BACKEND", "openai_compat")
import cv2
from simfoundry.models.vlm import Gemini

src = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/example_videos/PutCupInBowl.mp4"
out = Path("/data/projects/simfoundry/qwen_test"); out.mkdir(parents=True, exist_ok=True)

if src.lower().endswith((".mp4", ".mov", ".avi")):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read(); cap.release()
    assert ok, f"读不到视频帧：{src}"
else:
    frame = cv2.imread(src)
img_path = out / "frame.png"; cv2.imwrite(str(img_path), frame)
h, w = frame.shape[:2]

prompt = (
    "Detect every distinct object on the table. Return ONLY a JSON list, each item "
    '{"label": str, "box_2d": [ymin, xmin, ymax, xmax]} with coordinates normalized to 0-1000.'
)
vlm = Gemini(model="gemini-3.1-pro-preview", verbose=True)
res = vlm(prompt, image_paths=str(img_path))
text = vlm.get_result_text(res)
(out / "reply.txt").write_text(text)
print("---- 模型回复 ----\n", text)

m = re.search(r"\[.*\]", text, re.S)
items = json.loads(m.group(0)) if m else []
for it in items:
    y0, x0, y1, x1 = it["box_2d"]
    p0 = (int(x0 / 1000 * w), int(y0 / 1000 * h)); p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
    cv2.rectangle(frame, p0, p1, (0, 255, 0), 2)
    cv2.putText(frame, it.get("label", "?"), (p0[0], max(15, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(str(out / "boxes.png"), frame)
print(f"\n共 {len(items)} 个框，已画到 {out/'boxes.png'}，下载到 Mac 看框是否对准物体。")
