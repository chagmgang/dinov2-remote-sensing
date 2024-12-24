import argparse
import torch

from tqdm import tqdm
from dinov2.models import DINOv2
from dinov2.data import LinProbDataset, build_linprob_transforms
from sklearn.linear_model import LogisticRegression

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str)
parser.add_argument('--data-root', type=str)
parser.add_argument('--train-text', type=str)
parser.add_argument('--test-text', type=str)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num-workers', default=4, type=int)


def get_features(model, dataloader):

    all_features = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            outputs = model(data['pixel_values'].to(model.device))
            features = outputs.x_norm_clstoken
            all_features.append(features)
            all_labels.append(data['labels'])
    
    all_features = torch.cat(all_features).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return all_features, all_labels


def main():

    args = parser.parse_args()

    model = DINOv2.from_pretrained(
        args.model_path,
        device_map='cuda',
        torch_dtype=torch.float32,
    )
    model = model.eval()

    transform = build_linprob_transforms(
        img_size=model.config.img_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_dataset = LinProbDataset(
        data_root=args.data_root,
        filename=args.train_text,
        transform=transform,
    )

    test_dataset = LinProbDataset(
        data_root=args.data_root,
        filename=args.test_text,
        transform=transform,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    train_features, train_labels = get_features(model.student.backbone, train_dataloader)
    test_features, test_labels = get_features(model.student.backbone, test_dataloader)

    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

if __name__ == '__main__':
    main()
