import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torchmetrics.image.kid import KernelInceptionDistance
import random


@torch.no_grad()
def validate(model, val_loader, step, kid_subset_size=100):
    """
    Validation loop for UGATIT.
    Computes discriminator/gen losses, KID, and produces visualization-ready image grids.
    Returns:
        dict: {
            'val_disc_loss': float,
            'val_gen_loss': float,
            'KID': float,
            'images': {'RealA_FakeB': tensor, 'RealB_FakeA': tensor}
        }
    """
    import torch
    import random
    from torchvision.utils import make_grid
    from torchmetrics.image.kid import KernelInceptionDistance
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    val_gen_loss, val_disc_loss = 0.0, 0.0
    num_batches = 0

    # Initialize KID metric
    kid_metric = KernelInceptionDistance(subset_size=50).to(device)
    all_real_imgs, all_fake_imgs = [], []

    # Visualization buffers (first batch only)
    vis_real_A = vis_real_B = vis_fake_A = vis_fake_B = None

    pbar = tqdm(val_loader, desc=f"Validation @ step {step}")
    for batch in pbar:
        real_A = batch['real_A'].to(device, non_blocking=True)
        real_B = batch['real_B'].to(device, non_blocking=True)

        # --- Forward pass for losses ---
        d_loss, g_loss, _ = model(real_A, real_B)
        val_disc_loss += d_loss.mean().item()
        val_gen_loss += g_loss.mean().item()
        num_batches += 1

        # --- Generate fake images for metrics and visualization ---
        genA2B = model.module.genA2B if hasattr(model, "module") else model.genA2B
        genB2A = model.module.genB2A if hasattr(model, "module") else model.genB2A

        fake_B, _, _ = genA2B(real_A)
        fake_A, _, _ = genB2A(real_B)

        # --- Collect samples for KID (random subset to avoid OOM) ---
        if len(all_fake_imgs) < kid_subset_size and random.random() < 0.3:
            # Assumes input is normalized in [-1, 1]
            real_imgs = ((real_B * 0.5) + 0.5).clamp(0, 1).cpu()
            fake_imgs = ((fake_B * 0.5) + 0.5).clamp(0, 1).cpu()
            all_real_imgs.append(real_imgs)
            all_fake_imgs.append(fake_imgs)

        # --- Save first batch for visualization ---
        if vis_real_A is None:
            vis_real_A = real_A[:2].detach().cpu()
            vis_real_B = real_B[:2].detach().cpu()
            vis_fake_B = fake_B[:2].detach().cpu()
            vis_fake_A = fake_A[:2].detach().cpu()

    # --- Compute KID ---
    if len(all_fake_imgs) > 0:
        real_subset = torch.cat(all_real_imgs, dim=0)
        fake_subset = torch.cat(all_fake_imgs, dim=0)

        # Limit to subset size
        if len(real_subset) > kid_subset_size:
            idx = torch.randperm(len(real_subset))[:kid_subset_size]
            real_subset = real_subset[idx]
            fake_subset = fake_subset[idx]

        # KID expects uint8 images in [0, 255]
        real_subset = (real_subset * 255).to(torch.uint8)
        fake_subset = (fake_subset * 255).to(torch.uint8)

        kid_metric.update(real_subset.to(device), real=True)
        kid_metric.update(fake_subset.to(device), real=False)

        kid_mean, kid_std = kid_metric.compute()  # returns (mean, std)
        kid_score = kid_mean.item()
    else:
        kid_score = float("nan")

    # --- Restore training mode ---
    model.train()

    # --- Average losses ---
    avg_d = val_disc_loss / num_batches
    avg_g = val_gen_loss / num_batches

    # --- Make image grids for visualization (wandb-ready) ---
    image_grids = {}
    if vis_real_A is not None:
        # Normalize to [0,1] for visualization
        vis_real_A = ((vis_real_A * 0.5) + 0.5).clamp(0, 1)
        vis_fake_B = ((vis_fake_B * 0.5) + 0.5).clamp(0, 1)
        vis_real_B = ((vis_real_B * 0.5) + 0.5).clamp(0, 1)
        vis_fake_A = ((vis_fake_A * 0.5) + 0.5).clamp(0, 1)

        grid_A = make_grid(torch.cat([vis_real_A, vis_fake_B], dim=0), nrow=2)
        grid_B = make_grid(torch.cat([vis_real_B, vis_fake_A], dim=0), nrow=2)

        image_grids = {
            "RealA_FakeB": grid_A.cpu(),
            "RealB_FakeA": grid_B.cpu(),
        }

    print(
        f"🔍 Validation @ step {step}: "
        f"Disc={avg_d:.4f}, Gen={avg_g:.4f}, KID={kid_score:.4f}"
    )

    return {
        "val_disc_loss": avg_d,
        "val_gen_loss": avg_g,
        "KID": kid_score,
        "images": image_grids,
    }
