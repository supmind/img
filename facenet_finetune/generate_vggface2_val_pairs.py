import os
import argparse
import pandas as pd
from tqdm import tqdm
import random
from itertools import combinations

def generate_pairs(dataroot, num_same_pairs, num_diff_pairs):
    """
    Generates a list of same and different pairs from a dataset directory.
    Dataset is expected to be structured as:
    dataroot/
        person1/
            img1.jpg
            img2.jpg
        person2/
            img1.jpg
            ...
    """
    pairs = []

    # --- Step 1: Scan directory and map identities to images ---
    identity_map = {}
    for person_dir in tqdm(os.listdir(dataroot), desc="Scanning identities"):
        person_path = os.path.join(dataroot, person_dir)
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 1: # Need at least 2 images for a 'same' pair
                identity_map[person_dir] = images

    identities = list(identity_map.keys())
    if len(identities) < 2:
        raise ValueError("Need at least two different identities to generate pairs.")

    # --- Step 2: Generate 'same' pairs ---
    same_pairs_generated = 0
    pbar_same = tqdm(total=num_same_pairs, desc="Generating 'same' pairs")
    attempts = 0
    max_attempts = num_same_pairs * 5 # Avoid infinite loops

    while same_pairs_generated < num_same_pairs and attempts < max_attempts:
        person = random.choice(identities)
        img1, img2 = random.sample(identity_map[person], 2)

        # To avoid adding the same pair in different order
        sorted_pair = tuple(sorted((img1, img2)))
        if (sorted_pair[0], sorted_pair[1], 1) not in pairs:
            pairs.append((img1, img2, 1))
            same_pairs_generated += 1
            pbar_same.update(1)
        attempts += 1
    pbar_same.close()

    # --- Step 3: Generate 'different' pairs ---
    diff_pairs_generated = 0
    pbar_diff = tqdm(total=num_diff_pairs, desc="Generating 'different' pairs")
    attempts = 0
    max_attempts = num_diff_pairs * 5

    while diff_pairs_generated < num_diff_pairs and attempts < max_attempts:
        person1, person2 = random.sample(identities, 2)
        img1 = random.choice(identity_map[person1])
        img2 = random.choice(identity_map[person2])

        sorted_pair = tuple(sorted((img1, img2)))
        # Check to ensure this pair (or its reverse) isn't already a 'different' pair
        if (sorted_pair[0], sorted_pair[1], 0) not in pairs:
             pairs.append((img1, img2, 0))
             diff_pairs_generated += 1
             pbar_diff.update(1)
        attempts += 1
    pbar_diff.close()

    random.shuffle(pairs)
    return pairs


def main(args):
    """ Main function to generate the validation pairs CSV file. """
    print(f"Generating {args.num_pairs} 'same' and {args.num_pairs} 'different' pairs.")

    # We generate num_pairs for 'same' and num_pairs for 'different'
    total_pairs_to_generate = args.num_pairs

    pairs = generate_pairs(args.dataroot, total_pairs_to_generate, total_pairs_to_generate)

    df = pd.DataFrame(pairs, columns=["path1", "path2", "is_same"])

    output_dir = os.path.dirname(args.outfile)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(args.outfile, index=False)

    print(f"\nSuccessfully created validation pairs CSV at: {args.outfile}")
    print(f"Total pairs generated: {len(df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate validation pairs from a dataset directory.")

    parser.add_argument('--dataroot', type=str, required=True,
                        help="Path to the validation dataset directory.")
    parser.add_argument('--num_pairs', type=int, default=3000,
                        help="The number of 'same' and 'different' pairs to generate (e.g., 3000 means 3000 same and 3000 different).")
    parser.add_argument('--outfile', type=str, default='facenet_finetune/data/vggface2_val_pairs.csv',
                        help="Path to save the output CSV file.")

    cli_args = parser.parse_args()
    main(cli_args)

    # Example Usage:
    # python facenet_finetune/generate_vggface2_val_pairs.py --dataroot /path/to/vggface2/val
