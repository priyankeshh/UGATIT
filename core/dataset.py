from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as data

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)


class UGATITPairDataset(data.Dataset):
    """
    Combined dataset returning paired samples (real_A, real_B) for UGATIT or CycleGAN.
    Handles different dataset sizes by cycling through the smaller one.
    """

    def __init__(self, root_dir, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase

        # Define paths for domain A and B
        self.dir_A = os.path.join(root_dir, f"{phase}A")
        self.dir_B = os.path.join(root_dir, f"{phase}B")

        # Load datasets
        self.dataset_A = ImageFolder(self.dir_A, transform=self.transform)
        self.dataset_B = ImageFolder(self.dir_B, transform=self.transform)

        # Take max length so both domains are cycled evenly
        self.length = max(len(self.dataset_A), len(self.dataset_B))

        print(f"[UGATITPairDataset] Loaded {len(self.dataset_A)} images from {phase}A "
              f"and {len(self.dataset_B)} from {phase}B.")

    def __getitem__(self, index):
        # Cycle through smaller dataset
        img_A, _ = self.dataset_A[index % len(self.dataset_A)]
        img_B, _ = self.dataset_B[random.randint(0, len(self.dataset_B) - 1)]

        return {'real_A': img_A, 'real_B': img_B}

    def __len__(self):
        return self.length


def fetch_dataloader(cfg):
    """
    Create the UGATIT dataloader similar to RAFT-style structure.
    Returns a single loader yielding paired (real_A, real_B) batches.
    """

    # --------------------------
    # Define augmentations
    # --------------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((cfg.img_size + 30, cfg.img_size + 30)),
        transforms.RandomCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset_root = os.path.join('dataset', cfg.dataset)

    # --------------------------
    # Training dataloader
    # --------------------------
    if cfg.phase == 'train':
        train_dataset = UGATITPairDataset(
            dataset_root, transform=train_transform, phase='train')
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )

        print(f"[Data] Training with {len(train_dataset)} paired samples.")
        return train_loader

    # --------------------------
    # Test dataloader
    # --------------------------
    elif cfg.phase == 'test':
        test_dataset = UGATITPairDataset(
            dataset_root, transform=test_transform, phase='test')
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"[Data] Testing with {len(test_dataset)} paired samples.")
        return test_loader
