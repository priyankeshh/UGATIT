import torch
import torch.nn as nn

L1_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()
BCE_loss = nn.BCEWithLogitsLoss()


def adversarial_loss(D_real, D_fake, loss_fn):
    """Compute standard GAN loss for discriminator"""
    real_loss = loss_fn(D_real, torch.ones_like(D_real).to(D_real.device))
    fake_loss = loss_fn(D_fake, torch.zeros_like(D_fake).to(D_fake.device))
    return real_loss + fake_loss


def generator_adv_loss(D_fake, loss_fn):
    """Compute generator adversarial loss"""
    return loss_fn(D_fake, torch.ones_like(D_fake).to(D_fake.device))


def cycle_loss(fake2real, real, loss_fn):
    """Compute cycle-consistency loss"""
    return loss_fn(fake2real, real)


def identity_loss(fake_identity, real, loss_fn):
    """Compute identity loss"""
    return loss_fn(fake_identity, real)


def cam_loss(fake_translated_cam, fake_identity_cam, bce_fn):
    return bce_fn(fake_translated_cam, torch.ones_like(fake_translated_cam)) + \
        bce_fn(fake_identity_cam, torch.zeros_like(fake_identity_cam))
