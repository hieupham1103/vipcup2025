import cv2
import os
import shutil

# ==== Cấu hình đầu vào ====
split = 'test'
type_video = 'IR'
image_folder = f'/data4t/vipcup/hieu/vipcup_data/det/split_A/{type_video}/images/{split}'
label_folder = f'/data4t/vipcup/hieu/vipcup_data/det/split_A/{type_video}/labels/{split}'
output_folder = f'/data4t/vipcup/hieu/vipcup_data/track_video/split_A/{type_video}/{split}'
info_file = os.path.join(output_folder, 'info.txt')
step_size = 900
fps = 30

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# 1) Lấy danh sách tất cả ảnh trong image_folder
images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
print(f"Total images found: {len(images)}")

# 2) Chia thành các nhóm mỗi nhóm step_size ảnh
image_batches = [images[i:i + step_size] for i in range(0, len(images), step_size)]

# 3) Mở file info.txt để ghi thông tin
with open(info_file, 'w') as info_f:
    info_f.write("video_name,num_frames\n")

# 4) Duyệt từng batch, tạo video và copy nhãn tương ứng
for batch_idx, batch in enumerate(image_batches):
    # --- 4.1 Tạo tên video và đường dẫn nhãn ---
    video_name = f"video_{batch_idx}.mp4"
    video_path = os.path.join(output_folder, "videos", video_name)
    labels_subdir = os.path.join(output_folder, "labels", f"video_{batch_idx}")
    os.makedirs(os.path.join(output_folder, "videos"), exist_ok=True)
    os.makedirs(labels_subdir, exist_ok=True)

    # --- 4.2 Ghi video từ batch ảnh ---
    # Lấy kích thước của ảnh đầu tiên để tạo VideoWriter
    first_img_path = os.path.join(image_folder, batch[0])
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        raise RuntimeError(f"Không đọc được ảnh: {first_img_path}")
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_idx, img_name in enumerate(batch):
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            raise RuntimeError(f"Không đọc được ảnh: {img_path}")
        writer.write(frame)

        # --- 4.3 Copy nhãn tương ứng sang file mới ---
        # Tên file label gốc (theo stem của ảnh)
        stem = os.path.splitext(img_name)[0]
        src_label = os.path.join(label_folder, stem + '.txt')
        dst_label = os.path.join(labels_subdir, f"{frame_idx}.txt")
        # Nếu file nhãn gốc tồn tại, copy nội dung; nếu không, tạo file trống
        if os.path.exists(src_label):
            shutil.copyfile(src_label, dst_label)
        else:
            # Tạo file trống để tránh lỗi khi dataset yêu cầu tồn tại
            open(dst_label, 'w').close()

    writer.release()
    print(f"Created {video_path} with {len(batch)} frames, labels in {labels_subdir}")

    # --- 4.4 Ghi thông tin vào info.txt ---
    with open(info_file, 'a') as info_f:
        info_f.write(f"{video_name},{len(batch)}\n")
