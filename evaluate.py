import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torchmetrics.image.kid import KernelInceptionDistance
import random


@torch.no_grad()
def validate(model, val_loader, step, kid_subset_size=100):
    """
    Validation loop for UGATIT.
    Returns:
        dict: {
            'val_disc_loss': float,
            'val_gen_loss': float,
            'KID': float,
            'images': {'RealA_FakeB': grid_tensor, 'RealB_FakeA': grid_tensor}
        }
    """
    model.eval()
    val_gen_loss, val_disc_loss = 0.0, 0.0
    num_batches = 0

    # For KID computation
    kid_metric = KernelInceptionDistance(subset_size=50).cuda()
    all_real_imgs, all_fake_imgs = [], []

    # For visualization
    vis_real_A = vis_real_B = vis_fake_A = vis_fake_B = None

    pbar = tqdm(val_loader, desc=f"Validation @ step {step}")
    for batch in pbar:
        real_A = batch['real_A'].cuda(non_blocking=True)
        real_B = batch['real_B'].cuda(non_blocking=True)

        # Forward pass (for loss)
        d_loss, g_loss, _ = model(real_A, real_B)

        # Generate fake images explicitly
        fake_B, _, _ = model.module.genA2B(real_A)
        fake_A, _, _ = model.module.genB2A(real_B)

        val_disc_loss += d_loss.mean().item()
        val_gen_loss += g_loss.mean().item()
        num_batches += 1

        # Collect samples for KID
        if len(all_fake_imgs) < kid_subset_size and random.random() < 0.3:
            real_imgs = (real_B * 0.5 + 0.5).clamp(0, 1).cpu()
            fake_imgs = (fake_B * 0.5 + 0.5).clamp(0, 1).cpu()
            all_real_imgs.append(real_imgs)
            all_fake_imgs.append(fake_imgs)

        # Visualization (first batch only)
        if vis_real_A is None:
            vis_real_A = real_A[:2].cpu()
            vis_real_B = real_B[:2].cpu()
            vis_fake_B = fake_B[:2].cpu()
            vis_fake_A = fake_A[:2].cpu()

    # --- Compute KID ---
    if len(all_fake_imgs) > 0:
        real_subset = torch.cat(all_real_imgs, dim=0)
        fake_subset = torch.cat(all_fake_imgs, dim=0)
        if len(real_subset) > kid_subset_size:
            idx = torch.randperm(len(real_subset))[:kid_subset_size]
            real_subset = real_subset[idx]
            fake_subset = fake_subset[idx]
        kid_metric.update(real_subset.cuda(), real=True)
        kid_metric.update(fake_subset.cuda(), real=False)
        kid_score = kid_metric.compute().item()
    else:
        kid_score = float('nan')

    model.train()

    # --- Average losses ---
    avg_d = val_disc_loss / num_batches
    avg_g = val_gen_loss / num_batches

    # --- Make image grids ---
    image_grids = {}
    if vis_real_A is not None:
        grid_A = make_grid(
            torch.cat([vis_real_A, vis_fake_B], dim=0),
            nrow=2,
            normalize=True
        )
        grid_B = make_grid(
            torch.cat([vis_real_B, vis_fake_A], dim=0),
            nrow=2,
            normalize=True
        )
        image_grids = {'RealA_FakeB': grid_A, 'RealB_FakeA': grid_B}

    print(
        f"🔍 Validation @ step {step}: "
        f"Disc={avg_d:.4f}, Gen={avg_g:.4f}, KID={kid_score:.4f}"
    )

    return {
        'val_disc_loss': avg_d,
        'val_gen_loss': avg_g,
        'KID': kid_score,
        'images': image_grids
    }
