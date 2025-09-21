import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Import our custom model
from model import get_finetune_model

class VGGFace2Dataset(Dataset):
    """Custom VGGFace2 Dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a dataset, samples N classes (persons) and K samples (images) for each class.
    Returns batches of size N*K.
    """
    def __init__(self, dataset, n_classes, n_samples):
        super().__init__(None, batch_size=n_classes * n_samples, drop_last=False)
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.labels = self.dataset.dataframe['label'].values
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


def batch_hard_triplet_loss(labels, embeddings, margin, device):
    """
    Computes the batch-hard triplet loss.
    """
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    mask_anchor_positive = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(device)
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    mask_anchor_negative = (labels.unsqueeze(1) != labels.unsqueeze(0)).float().to(device)
    # Add a large value to the diagonal and where we have positives
    max_dist = torch.max(pairwise_dist).item()
    anchor_negative_dist = pairwise_dist + max_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    triplet_loss = torch.relu(hardest_positive_dist - hardest_negative_dist + margin)
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # --- Data Preparation ---
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = VGGFace2Dataset(csv_file=args.csv_path, transform=transform)

    # Use our custom sampler
    batch_sampler = BalancedBatchSampler(dataset, n_classes=args.classes_per_batch, n_samples=args.samples_per_class)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)

    # --- Model, Optimizer, Loss ---
    model = get_finetune_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # We implement the loss function manually for batch-hard mining
    # triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)

    # --- Training Loop ---
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            embeddings = model(images)

            loss = batch_hard_triplet_loss(labels, embeddings, args.margin, device)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")

    # --- Save Model ---
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    torch.save(model.state_dict(), args.save_path)
    print(f"Finetuned model saved to {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune Facenet model with 128d embeddings")

    parser.add_argument('--csv_path', type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument('--save_path', type=str, default='models/facenet_128d.pt', help="Path to save the finetuned model.")

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--classes_per_batch', type=int, default=32, help="Number of distinct classes (persons) per batch.")
    parser.add_argument('--samples_per_class', type=int, default=4, help="Number of samples (images) per class in a batch.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--margin', type=float, default=0.5, help="Margin for the triplet loss.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for the DataLoader.")

    cli_args = parser.parse_args()

    # batch_size is implicit: classes_per_batch * samples_per_class
    print(f"Batch size: {cli_args.classes_per_batch * cli_args.samples_per_class}")

    main(cli_args)

    # Example usage:
    # python facenet_finetune/finetune_face.py --csv_path facenet_finetune/data/vggface2_train.csv
