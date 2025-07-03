
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src import BackgroundSubtractorModule

# ==== THAY CÁC BIẾN ĐẦU VÀO Ở ĐÂY ====
split = 'train'
type_video = 'IR'
video_dir = Path(f"data/track_video/split_A/{type_video}/{split}/videos")
label_dir = Path(f"data/track_video/split_A/{type_video}/{split}/labels")
out_img_dir = Path(f"data/track_video/split_A/{type_video}/{split}/train_images")
out_label_dir = Path(f"data/track_video/split_A/{type_video}/{split}/train_labels")

window_size = 15
gray_img = True
image_size = (256, 256)  # (H, W)
heatmap_sigma = 10.0  # Không dùng ở đây
fps = 30  # Không dùng ở đây
# ====================================


def load_video_frames(video_path: Path, resize_size: tuple, gray: bool):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    H, W = resize_size
    to_tensor = transforms.ToTensor()

    frame_list = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if gray:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("L")
        else:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

        pil = pil.resize((W, H))
        t = to_tensor(pil)
        frame_list.append(t)

    cap.release()

    if len(frame_list) == 0:
        raise RuntimeError(f"No frames found in {video_path}")

    return torch.stack(frame_list, dim=0)


def main():
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    )
    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    device = torch.device("cpu")

    bgsub_mod = BackgroundSubtractorModule(
        window_size=window_size,
        gray_img=gray_img,
        eps=1e-5,
        normalize_output=True
    ).to(device)
    bgsub_mod.eval()

    for vid_path in video_files:
        video_name = vid_path.stem
        print(f"\n=== Processing video: {video_name} ===")

        video_label_subdir = label_dir / video_name
        if not video_label_subdir.exists() or not video_label_subdir.is_dir():
            print(f"  Warning: label folder {video_label_subdir} does not exist. Skipping.")
            continue

        print("  Loading frames into RAM…")
        frames_tensor = load_video_frames(
            video_path=vid_path,
            resize_size=image_size,
            gray=gray_img
        )
        T, C, H, W = frames_tensor.shape
        print(f"    {T} frames loaded. Tensor shape = {frames_tensor.shape}")

        print("  Running BackgroundSubtractorModule…")
        with torch.no_grad():
            frames_bgsub = bgsub_mod(frames_tensor.to(device))
        frames_bgsub = frames_bgsub.cpu()
        C_out = frames_bgsub.shape[1]
        print(f"    Background-subtracted frames shape = {frames_bgsub.shape}")

        this_out_img_subdir = out_img_dir
        this_out_label_subdir = out_label_dir
        this_out_img_subdir.mkdir(parents=True, exist_ok=True)
        this_out_label_subdir.mkdir(parents=True, exist_ok=True)

        for i in range(T):
            frame_tensor = frames_bgsub[i]
            np_frame = (frame_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            if C_out == 1:
                np_frame = np_frame.squeeze(-1)

            frame_name = f"{video_name}_frame_{i:04d}.jpg"
            cv2.imwrite(str(this_out_img_subdir / frame_name), np_frame)

            src_label = video_label_subdir / f"{i}.txt"
            dst_label = this_out_label_subdir / f"{video_name}_frame_{i:04d}.txt"
            if src_label.exists():
                shutil.copyfile(str(src_label), str(dst_label))
            else:
                open(dst_label, "w").close()

        print(f"  Saved {T} frames and labels under:")
        print(f"    Images: {this_out_img_subdir}")
        print(f"    Labels: {this_out_label_subdir}")

    print("\nAll videos processed. Exiting.")


if __name__ == "__main__":
    main()