import cv2
import os
import numpy as np

def add_label(frame, label, width, label_height=50):
    """Thêm nhãn phía trên frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size

    # Tạo canvas mới có thêm vùng label ở trên
    labeled_frame = np.zeros((frame.shape[0] + label_height, width, 3), dtype=np.uint8)
    labeled_frame[label_height:, :, :] = frame
    # Vẽ chữ giữa khung
    x = (width - text_w) // 2
    y = (label_height + text_h) // 2
    cv2.putText(labeled_frame, label, (x, y), font, font_scale, (255, 255, 255), thickness)

    return labeled_frame

def merge_videos_side_by_side(video_paths, output_path="merged.mp4"):
    # 1. Mở VideoCapture và kiểm tra
    caps = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"[ERROR] Không mở được video: {p}")
            return
        caps.append(cap)
    print("[INFO] Tất cả video đã được mở thành công.")

    # 2. Lấy FPS và kích thước:
    fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    fps = int(min(fps_list))
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
    widths  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  for cap in caps]
    min_h = max(heights)
    # Tính lại width theo tỷ lệ
    resized_widths = [int(w * (min_h / h)) for w, h in zip(widths, heights)]
    total_w = sum(resized_widths)
    label_h = 20
    out_size = (total_w, min_h + label_h)
    print(f"[INFO] FPS={fps}, output size={out_size}")

    # 3. Tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    # 4. Đọc frame, resize, thêm label, ghép cạnh nhau
    labels = [os.path.basename(p) for p in video_paths]
    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] Video thứ {i} ({labels[i]}) đã kết thúc.")
                cap.release()
                # Nếu muốn dừng khi tất cả hết, bỏ return, dùng break và track count.
                out.release()
                return
            # resize theo chiều cao
            frame = cv2.resize(frame, (resized_widths[i], min_h))
            labeled = add_label(frame, labels[i], resized_widths[i], label_height=label_h)
            frames.append(labeled)

        merged = np.hstack(frames)
        out.write(merged)

    # 5. Giải phóng nếu vòng while kết thúc
    out.release()
    print(f"[INFO] Đã ghi xong: {output_path}")

if __name__ == "__main__":
    # video_paths = [
    #     "outputs/8n_CT_0.25.mp4",
    #     "outputs/8n_CT_0.3.mp4",
    #     "outputs/8n_CT_0.35.mp4",
    #     "outputs/8n_CT_0.4.mp4",
    #     "outputs/8n_CT_0.45.mp4",
    #     "outputs/8n_CT_0.5.mp4",
    # ]
    video_paths = [
        "outputs/8n_0.3.mp4",
        "outputs/8n_CT_0.3.mp4",
    ]
    
    merge_videos_side_by_side(video_paths, output_path="outputs/merged_video.mp4")
