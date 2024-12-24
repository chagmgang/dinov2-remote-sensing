import os
import torch
import torchvision
from PIL import Image


class LinProbDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, filename, transform):

        self.data_root = data_root
        self.images, self.labels = self.parse_filename(filename)
        self.transform = transform

        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_root,
            self.images[idx],
        )
        label = self.labels[idx]

        image = Image.open(filename).convert('RGB')
        return {
            'pixel_values': self.transform(image),
            'labels': label,
        }

    def __len__(self):
        return len(self.images)

    def parse_filename(self, filename):
        f = open(filename, 'r')
        images, labels = list(), list()
        while True:
            line = f.readline()
            if not line: break
            line = line.replace('\n', '')
            image, label = line.split(',')
            images.append(image)
            labels.append(int(label))
        f.close()
        return images, labels


def build_linprob_transforms(img_size, mean, std):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    return transforms
