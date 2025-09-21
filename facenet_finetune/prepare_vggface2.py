import os
import pandas as pd
import argparse
from tqdm import tqdm

def create_dataset_index(dataset_path, output_path):
    """
    Scans the VGGFace2 dataset directory and creates a CSV index file.

    The VGGFace2 dataset is expected to have a structure like:
    .../train/
        n000001/
            0001_01.jpg
            0002_01.jpg
            ...
        n000002/
            0001_01.jpg
            ...

    Args:
        dataset_path (str): The path to the VGGFace2 'train' directory.
        output_path (str): The path where the output CSV file will be saved.
    """
    print(f"Scanning dataset directory: {dataset_path}")

    # Get all identity folder names
    identities = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    identities.sort()

    # Create a mapping from identity folder name to a unique integer label
    label_map = {identity: i for i, identity in enumerate(identities)}

    records = []
    # Use tqdm for a progress bar
    for identity_name in tqdm(identities, desc="Processing identities"):
        class_id = label_map[identity_name]
        identity_path = os.path.join(dataset_path, identity_name)

        # List all image files for the current identity
        image_files = [f for f in os.listdir(identity_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_name in image_files:
            image_path = os.path.join(identity_path, image_name)
            records.append({
                "path": os.path.abspath(image_path),
                "label": class_id
            })

    if not records:
        print("Warning: No images found. Please check the dataset path and structure.")
        return

    # Create a pandas DataFrame
    df = pd.DataFrame(records)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Successfully created dataset index at: {output_path}")
    print(f"Total identities found: {len(identities)}")
    print(f"Total images found: {len(df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare VGGFace2 dataset by creating an index file.")
    parser.add_argument('--dataroot', type=str, required=True,
                        help="Path to the root of the VGGFace2 'train' directory.")
    parser.add_argument('--outfile', type=str, default='data/vggface2_train.csv',
                        help="Path to save the output CSV file.")

    args = parser.parse_args()

    # The output file path should be relative to the script's directory if not absolute
    script_dir = os.path.dirname(__file__)
    output_file = args.outfile
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)


    create_dataset_index(args.dataroot, output_file)

    # Example usage:
    # python facenet_finetune/prepare_vggface2.py --dataroot /path/to/vggface2/train
    # This will generate facenet_finetune/data/vggface2_train.csv
