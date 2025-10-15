import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm
import random

@torch.no_grad()
def validate(model, val_loader, step, kid_subset_size=100, kid_metric_subset=50):
    device = next(model.parameters()).device
    model.eval()

    val_gen_loss, val_disc_loss, num_batches = 0.0, 0.0, 0
    kid_metric = KernelInceptionDistance(subset_size=kid_metric_subset, normalize=False).to(device)
    all_real_imgs, all_fake_imgs = [], []

    vis_real_A = vis_real_B = vis_fake_A = vis_fake_B = None
    target_hw = None
    inception_hw = (299, 299)
    total_batches = len(val_loader)
    vis_targets = random.sample(range(total_batches), 2) if total_batches >= 2 else [0, 0]

    pbar = tqdm(val_loader, desc=f"Validation @ step {step}")
    for i, batch in enumerate(pbar):
        real_A = batch['real_A'].to(device)
        real_B = batch['real_B'].to(device)

        d_loss, g_loss, _ = model(real_A, real_B)
        val_disc_loss += float(d_loss.mean().item())
        val_gen_loss += float(g_loss.mean().item())
        num_batches += 1

        genA2B = model.module.genA2B if hasattr(model, "module") else model.genA2B
        genB2A = model.module.genB2A if hasattr(model, "module") else model.genB2A
        out_B, out_A = genA2B(real_A), genB2A(real_B)
        fake_B = out_B[0] if isinstance(out_B, (tuple, list)) else out_B
        fake_A = out_A[0] if isinstance(out_A, (tuple, list)) else out_A

        if target_hw is None:
            target_hw = (real_B.shape[-2], real_B.shape[-1])

        if len(all_fake_imgs) < kid_subset_size and random.random() < 0.3:
            real_imgs = ((real_B * 0.5) + 0.5).clamp(0, 1)
            fake_imgs = ((fake_B * 0.5) + 0.5).clamp(0, 1)
            if real_imgs.shape[-2:] != target_hw:
                real_imgs = F.interpolate(real_imgs, size=target_hw, mode='bilinear', align_corners=False)
                fake_imgs = F.interpolate(fake_imgs, size=target_hw, mode='bilinear', align_corners=False)
            all_real_imgs.append(real_imgs.detach().cpu())
            all_fake_imgs.append(fake_imgs.detach().cpu())

        if i in vis_targets:
            if vis_real_A is None:
                vis_real_A, vis_real_B = real_A.cpu(), real_B.cpu()
                vis_fake_B, vis_fake_A = fake_B.cpu(), fake_A.cpu()
            else:
                vis_real_A = torch.cat([vis_real_A, real_A.cpu()], dim=0)
                vis_real_B = torch.cat([vis_real_B, real_B.cpu()], dim=0)
                vis_fake_B = torch.cat([vis_fake_B, fake_B.cpu()], dim=0)
                vis_fake_A = torch.cat([vis_fake_A, fake_A.cpu()], dim=0)

    if len(all_fake_imgs) > 0:
        real_subset = torch.cat(all_real_imgs, dim=0)
        fake_subset = torch.cat(all_fake_imgs, dim=0)
        if len(real_subset) > kid_subset_size:
            idx = torch.randperm(len(real_subset))[:kid_subset_size]
            real_subset, fake_subset = real_subset[idx], fake_subset[idx]
        real_subset = F.interpolate(real_subset, size=inception_hw, mode='bilinear', align_corners=False)
        fake_subset = F.interpolate(fake_subset, size=inception_hw, mode='bilinear', align_corners=False)
        real_uint8 = (real_subset * 255.0).round().clamp(0, 255).to(torch.uint8).to(device)
        fake_uint8 = (fake_subset * 255.0).round().clamp(0, 255).to(torch.uint8).to(device)
        kid_metric.update(real_uint8, real=True)
        kid_metric.update(fake_uint8, real=False)
        kid_mean, kid_std = kid_metric.compute()
        kid_score = float(kid_mean.item())
    else:
        kid_score = float("nan")

    model.train()
    avg_d, avg_g = val_disc_loss / max(1, num_batches), val_gen_loss / max(1, num_batches)

    image_grids = {}
    if vis_real_A is not None:
        vis_real_A = ((vis_real_A * 0.5) + 0.5).clamp(0, 1)
        vis_fake_B = ((vis_fake_B * 0.5) + 0.5).clamp(0, 1)
        vis_real_B = ((vis_real_B * 0.5) + 0.5).clamp(0, 1)
        vis_fake_A = ((vis_fake_A * 0.5) + 0.5).clamp(0, 1)
        if vis_real_A.shape[-2:] != target_hw:
            vis_real_A = F.interpolate(vis_real_A, size=target_hw, mode='bilinear', align_corners=False)
            vis_fake_B = F.interpolate(vis_fake_B, size=target_hw, mode='bilinear', align_corners=False)
            vis_real_B = F.interpolate(vis_real_B, size=target_hw, mode='bilinear', align_corners=False)
            vis_fake_A = F.interpolate(vis_fake_A, size=target_hw, mode='bilinear', align_corners=False)
        grid_A = make_grid(torch.cat([vis_real_A, vis_fake_B], dim=0), nrow=2)
        grid_B = make_grid(torch.cat([vis_real_B, vis_fake_A], dim=0), nrow=2)
        image_grids = {"RealA_FakeB": grid_A.cpu(), "RealB_FakeA": grid_B.cpu()}

    print(f"🔍 Validation @ step {step}: Disc={avg_d:.4f}, Gen={avg_g:.4f}, KID={kid_score:.4f}")
    return {
        "val_disc_loss": avg_d,
        "val_gen_loss": avg_g,
        "KID": kid_score * 100,
        "images": image_grids,
    }
