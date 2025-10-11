from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger
from scipy import misc
import os
import cv2
import torch
import numpy as np


def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img


def preprocessing(x):
    x = x/127.5 - 1  # -1 ~ 1
    return x


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(
        0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(
        0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


class Logger:
    def __init__(self, model, cfg):
        self.model = model
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.cfg = cfg

    def _init_writer(self):
        if self.writer is None:
            if self.cfg.log_dir is None:
                self.writer = SummaryWriter()
            else:
                self.writer = SummaryWriter(self.cfg.log_dir)

    def _print_training_status(self):
        # Average losses
        averaged_metrics = {
            k: self.running_loss[k] / self.cfg.sum_freq
            for k in sorted(self.running_loss.keys())
        }

        # Print nicely formatted metrics
        training_str = f"[Step {self.total_steps + 1:6d}]"
        metrics_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in averaged_metrics.items()])
        loguru_logger.info(training_str + " | " + metrics_str)
        print(training_str + " | " + metrics_str)

        # Write to TensorBoard
        self._init_writer()
        for k, v in averaged_metrics.items():
            self.writer.add_scalar(k, v, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key, value in metrics.items():
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += value

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        self._init_writer()
        for key, value in results.items():
            self.writer.add_scalar(key, value, self.total_steps)

    def close(self):
        if self.writer:
            self.writer.close()
