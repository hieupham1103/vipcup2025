import os



def count_obj(txt_file: str):
    paths = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            paths.append(line)
    return paths

label_folder = "/home/cvpr2025/yolo_testing/data/vipcup_det/split_A/RGB/labels/test/"

for file in os.listdir(label_folder):
    # print(file)
    if file.endswith('.txt'):
        objs = count_obj(label_folder+file)
        if len(objs) >= 2:
            print(f"{file}: {len(objs)} objects")