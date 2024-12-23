from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None

class BaseDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        filenames,
        transforms,
    ):
        self.filenames = filenames
        self.transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        data = self.transforms(image)
        image.close()
        return data, None

    def __len__(self):
        return len(self.filenames)
