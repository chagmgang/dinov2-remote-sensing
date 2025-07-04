import argparse
import torch
import faiss

from collections import Counter
from tqdm import tqdm
from dinov2.models import DINOv2
from dinov2.data import LinProbDataset, build_linprob_transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch.distributed as dist

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--local-rank', '--local_rank')
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
            features = outputs.x_norm_clstoken.cpu()
            all_features.append(features)
            all_labels.append(data['labels'])

    all_features = torch.cat(all_features).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return all_features, all_labels


def knn_faiss(train_features, train_labels, test_features, test_labels, k):
    index = faiss.IndexFlatL2(train_features.shape[-1])
    index.add(train_features)
    knn_dist, knn_idx = index.search(test_features, k=k)
    preds = train_labels[knn_idx]
    preds = np.array([np.bincount(pred).argmax() for pred in preds])
    accuracy = (preds == test_labels).mean() * 100.0
    return accuracy


def linprob(train_features, train_labels, test_features, test_labels):
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    return accuracy


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
    features = np.concatenate([train_features, test_features], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    num_classes = list(set(labels))
    # knn start
    for k in [1, 2, 5, 10, 20]:
        for train_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            accs = list()
            for iteration in range(10):
                train_features, test_features, train_labels, test_labels = train_test_split(
                    features,
                    labels,
                    test_size=(1 - train_ratio),
                    stratify=labels,
                )

                train_counter = Counter(train_labels)
                valid_eval = False
                for key in train_counter.keys():
                    if train_counter[key] > k:
                        valid_eval = True

                if valid_eval:
                    accuracy = knn_faiss(train_features, train_labels, test_features, test_labels, k)
                    accs.append(accuracy)

            if valid_eval:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f"k = {k}, train_ratio = {train_ratio}, iteration = {iteration}, Acc = {mean_acc:.3f} std = {std_acc:.3f}")

    # linear probing start
    for train_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        accs = list()
        for iteration in range(10):
            train_features, test_features, train_labels, test_labels = train_test_split(
                features,
                labels,
                test_size=(1 - train_ratio),
                stratify=labels,
            )
            accuracy = linprob(train_features, train_labels, test_features, test_labels)
            accs.append(accuracy)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"train_ratio = {train_ratio}, iteration = {iteration}, Acc = {mean_acc:.3f} std = {std_acc:.3f}")


if __name__ == '__main__':
    main()
