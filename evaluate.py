import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
import random

@torch.no_grad()
def validate(model, val_loader, step, kid_subset_size=100, kid_metric_subset=50):
    """
    Validation loop for UGATIT that is robust to generator return types and image sizes.

    Args:
        model: nn.Module (may be wrapped in DataParallel/DistributedDataParallel)
        val_loader: DataLoader yielding dicts with 'real_A' and 'real_B' (tensors in [-1,1])
        step: int (current training step, used only for display)
        kid_subset_size: int, number of images to collect for KID (total)
        kid_metric_subset: int, subset_size passed to KernelInceptionDistance

    Returns:
        dict with keys:
            'val_disc_loss', 'val_gen_loss', 'KID', 'images' (map of name -> CxHxW float tensor [0,1])
    """
    device = next(model.parameters()).device
    model.eval()

    val_gen_loss = 0.0
    val_disc_loss = 0.0
    num_batches = 0

    # KID metric: accepts subset_size parameter
    kid_metric = KernelInceptionDistance(subset_size=kid_metric_subset).to(device)

    # Buffers for KID: lists of tensors (C,H,W) in [0,1] on CPU
    all_real_imgs = []
    all_fake_imgs = []

    # Visualization - store first batch's first two examples
    vis_real_A = vis_real_B = vis_fake_A = vis_fake_B = None

    # to ensure consistent appended size, remember a target (H, W) - set on first batch
    target_hw = None

    pbar = tqdm(val_loader, desc=f"Validation @ step {step}")
    for batch in pbar:
        real_A = batch['real_A'].to(device, non_blocking=True)  # shape: (B, C, H, W), assumed in [-1,1]
        real_B = batch['real_B'].to(device, non_blocking=True)

        # Forward to compute losses (assumes model.forward returns D_loss, G_loss, metrics)
        d_loss, g_loss, _ = model(real_A, real_B)
        val_disc_loss += float(d_loss.mean().item())
        val_gen_loss += float(g_loss.mean().item())
        num_batches += 1

        # Safe access to generator functions (handles DataParallel / DDP wrappers)
        genA2B = model.module.genA2B if hasattr(model, "module") else model.genA2B
        genB2A = model.module.genB2A if hasattr(model, "module") else model.genB2A

        # Call generators and unpack outputs robustly
        out_B = genA2B(real_A)  # expected (fake_B, cam_logit_B, heatmap_B)
        out_A = genB2A(real_B)

        # Extract first element if a tuple/list; else assume it's already a tensor
        fake_B = out_B[0] if isinstance(out_B, (tuple, list)) else out_B
        fake_A = out_A[0] if isinstance(out_A, (tuple, list)) else out_A

        # Determine target spatial size once (use real_B's HxW)
        if target_hw is None:
            target_hw = (real_B.shape[-2], real_B.shape[-1])  # (H, W)

        # Collect subset for KID randomly, but bounded by kid_subset_size
        # Convert to [0,1] floats on CPU and ensure shapes are consistent before append
        if len(all_fake_imgs) < kid_subset_size and random.random() < 0.3:
            # Convert from [-1,1] -> [0,1]
            real_imgs = ((real_B * 0.5) + 0.5).clamp(0, 1)
            fake_imgs = ((fake_B * 0.5) + 0.5).clamp(0, 1)

            # Resize to target_hw if needed (bilinear for real/fake)
            if real_imgs.shape[-2:] != target_hw:
                real_imgs = F.interpolate(real_imgs, size=target_hw, mode='bilinear', align_corners=False)
            if fake_imgs.shape[-2:] != target_hw:
                fake_imgs = F.interpolate(fake_imgs, size=target_hw, mode='bilinear', align_corners=False)

            # detach -> cpu
            real_cpu = real_imgs.detach().cpu()
            fake_cpu = fake_imgs.detach().cpu()

            # Only append pure tensors
            if isinstance(real_cpu, torch.Tensor) and isinstance(fake_cpu, torch.Tensor):
                all_real_imgs.append(real_cpu)
                all_fake_imgs.append(fake_cpu)

        # Save first-batch images for visualization (use first 2 examples)
        if vis_real_A is None:
            vis_real_A = real_A[:2].detach().cpu()
            vis_real_B = real_B[:2].detach().cpu()
            vis_fake_B = fake_B[:2].detach().cpu()
            vis_fake_A = fake_A[:2].detach().cpu()

    # --- Compute KID if we collected anything ---
    if len(all_fake_imgs) > 0:
        # concat along batch dim
        real_subset = torch.cat(all_real_imgs, dim=0)  # shape (N, C, H, W)
        fake_subset = torch.cat(all_fake_imgs, dim=0)

        # If more than kid_subset_size, randomly subsample
        if len(real_subset) > kid_subset_size:
            idx = torch.randperm(len(real_subset))[:kid_subset_size]
            real_subset = real_subset[idx]
            fake_subset = fake_subset[idx]

        # KID metric expects uint8 images in [0,255]
        real_uint8 = (real_subset * 255.0).round().clamp(0, 255).to(torch.uint8).to(device)
        fake_uint8 = (fake_subset * 255.0).round().clamp(0, 255).to(torch.uint8).to(device)

        # Update and compute: .compute() returns (mean, std)
        kid_metric.update(real_uint8, real=True)
        kid_metric.update(fake_uint8, real=False)
        kid_mean, kid_std = kid_metric.compute()  # both are tensors (scalars)
        kid_score = float(kid_mean.item())
    else:
        kid_score = float("nan")

    # Restore model to train mode
    model.train()

    # Average losses
    avg_d = val_disc_loss / max(1, num_batches)
    avg_g = val_gen_loss / max(1, num_batches)

    # --- Build visualization grids (float tensors in [0,1]) ---
    image_grids = {}
    if vis_real_A is not None:
        # Convert from [-1,1] to [0,1], clamp
        vis_real_A = ((vis_real_A * 0.5) + 0.5).clamp(0, 1)
        vis_fake_B = ((vis_fake_B * 0.5) + 0.5).clamp(0, 1)
        vis_real_B = ((vis_real_B * 0.5) + 0.5).clamp(0, 1)
        vis_fake_A = ((vis_fake_A * 0.5) + 0.5).clamp(0, 1)

        # Ensure same spatial size
        if vis_real_A.shape[-2:] != target_hw:
            vis_real_A = F.interpolate(vis_real_A, size=target_hw, mode='bilinear', align_corners=False)
            vis_fake_B = F.interpolate(vis_fake_B, size=target_hw, mode='bilinear', align_corners=False)
            vis_real_B = F.interpolate(vis_real_B, size=target_hw, mode='bilinear', align_corners=False)
            vis_fake_A = F.interpolate(vis_fake_A, size=target_hw, mode='bilinear', align_corners=False)

        grid_A = make_grid(torch.cat([vis_real_A, vis_fake_B], dim=0), nrow=2)  # C x H x W
        grid_B = make_grid(torch.cat([vis_real_B, vis_fake_A], dim=0), nrow=2)

        # Keep on CPU float tensors in [0,1] for wandb
        image_grids = {
            "RealA_FakeB": grid_A.cpu(),
            "RealB_FakeA": grid_B.cpu(),
        }

    print(
        f"🔍 Validation @ step {step}: Disc={avg_d:.4f}, Gen={avg_g:.4f}, KID={kid_score:.4f}"
    )

    return {
        "val_disc_loss": avg_d,
        "val_gen_loss": avg_g,
        "KID": kid_score * 100,
        "images": image_grids,
    }
