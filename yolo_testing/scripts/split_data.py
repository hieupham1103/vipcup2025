import os
import random
import argparse

def list_images(directory, exts={'.jpg', '.jpeg', '.png', '.bmp'}):
    """
    Recursively list image files in the given directory.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                files.append(os.path.join(root, fname))
    return files


def write_list(file_list, output_path):
    """
    Write full paths of files in file_list to output_path, one per line.
    """
    with open(output_path, 'w') as f:
        for path in file_list:
            f.write(f"{path}\n")


def generate_data_yaml(output_path, base_path, train_txt, val_txt, test_txt, nc, names):
    """
    Generate a data.yml file for YOLOv10 with text lists.
    """
    lines = [
        f"path: {base_path}",
        f"train: {train_txt}",
        f"val:   {val_txt}",
        f"test:  {test_txt}\n",
        f"nc: {nc}",
        f"names: {names}"
    ]
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset and generate data.yml")
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path (same as existing `path` in old config)')
    parser.add_argument('--train_dir', type=str, default='images/train',
                        help='Relative train images folder')
    parser.add_argument('--test_dir', type=str, default='images/test',
                        help='Relative test images folder')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of train images to use for validation')
    parser.add_argument('--nc', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--names', nargs='+', required=True,
                        help='List of class names')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to write txt lists and data.yml')
    args = parser.parse_args()

    # Paths
    train_folder = os.path.join(args.base_path, args.train_dir)
    test_folder = os.path.join(args.base_path, args.test_dir)
    
    # List images
    train_images = list_images(train_folder)
    test_images = list_images(test_folder)

    # Split train into train/val
    random.shuffle(train_images)
    n_val = int(len(train_images) * args.val_ratio)
    val_images = train_images[:n_val]
    train_images = train_images[n_val:]

    # Prepare output paths
    train_txt = os.path.join(args.output_dir, 'train.txt')
    val_txt   = os.path.join(args.output_dir, 'val.txt')
    test_txt  = os.path.join(args.output_dir, 'test.txt')
    data_yaml = os.path.join(args.output_dir, 'data.yml')

    # Write lists
    write_list(train_images, train_txt)
    write_list(val_images, val_txt)
    write_list(test_images, test_txt)

    # Generate config
    generate_data_yaml(data_yaml, args.base_path, train_txt, val_txt, test_txt,
                       args.nc, args.names)

    print(f"Written train.txt ({len(train_images)} entries), val.txt ({len(val_images)} entries), test.txt ({len(test_images)} entries), and data.yml to {args.output_dir}")

if __name__ == '__main__':
    main()