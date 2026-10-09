#!/usr/bin/env python3
"""
测试本地 FLUX.1-Kontext：从示例视频取一帧，缩放到 FLUX 支持的分辨率，让它"去掉桌上的杯子"，
用来确认 FLUX 权重完整、能加载、能出图（s5/s6 会用同一个 FLUX1 类）。
用法：
  cd /data/SimFoundry && conda activate simfoundry
  python scripts/tools/test_flux_kontext.py docs/assets/example_videos/PutCupInBowl.mp4
  python scripts/tools/test_flux_kontext.py <视频或图片> "Remove the pen from the table."
输出：/data/projects/simfoundry/flux_test/input.png、output.png
"""
import os, sys, time
from pathlib import Path

os.environ.setdefault("SIMFOUNDRY_FLUX_KONTEXT_PATH", "/data/models/FLUX.1-Kontext-dev")
import cv2
import torch
from simfoundry.models.vlm import FLUX1, PREFERRED_KONTEXT_RESOLUTIONS

src = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/example_videos/PutCupInBowl.mp4"
prompt = sys.argv[2] if len(sys.argv) > 2 else (
    "Remove the white paper cup from the table. Keep everything else unchanged and fill in the wood table naturally."
)
out = Path("/data/projects/simfoundry/flux_test"); out.mkdir(parents=True, exist_ok=True)

if src.lower().endswith((".mp4", ".mov", ".avi")):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read(); cap.release()
    assert ok, f"读不到视频帧：{src}"
else:
    frame = cv2.imread(src)

# FLUX1.__call__ 只接受 PREFERRED_KONTEXT_RESOLUTIONS 里的尺寸，选宽高比最接近的一档。
h, w = frame.shape[:2]
tw, th = min(PREFERRED_KONTEXT_RESOLUTIONS, key=lambda r: abs(r[0] / r[1] - w / h))
frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
in_path = out / "input.png"; cv2.imwrite(str(in_path), frame)
print(f"输入帧 {w}x{h} -> {tw}x{th}")

t0 = time.time()
model = FLUX1(model="FLUX.1-Kontext-dev", dtype=torch.bfloat16, device="cuda")
print(f"模型加载完成，用时 {time.time() - t0:.0f}s")

t0 = time.time()
image = model(image_path=str(in_path), prompt=prompt, guidance_scale=2.5,
              num_inference_steps=20, max_sequence_length=512, seed=0)
image.save(out / "output.png")
print(f"出图完成，用时 {time.time() - t0:.0f}s，峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print(f"已保存 {out/'output.png'}，下载到 Mac 看杯子是否被去掉。")
