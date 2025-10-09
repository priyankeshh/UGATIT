import itertools
from core.networks import ResnetGenerator, Discriminator, RhoClipper
import torch
import torch.nn as nn
from core.loss import generator_adv_loss, cycle_loss, identity_loss, cam_loss, adversarial_loss


class UGATIT(nn.Module):
    def build_model(self, cfg):
        """Build U-GAT-IT model components"""

        self.cfg = cfg
        device = cfg.device
        img_size = cfg.img_size

        # ======================================================
        # Generator and Discriminator Setup
        # ======================================================
        ch = cfg.ch
        n_res = cfg.n_res
        light = cfg.light

        self.genA2B = ResnetGenerator(
            input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res,
            img_size=img_size, light=light
        ).to(device)

        self.genB2A = ResnetGenerator(
            input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res,
            img_size=img_size, light=light
        ).to(device)

        self.disGA = Discriminator(input_nc=3, ndf=ch, n_layers=7).to(device)
        self.disGB = Discriminator(input_nc=3, ndf=ch, n_layers=7).to(device)
        self.disLA = Discriminator(input_nc=3, ndf=ch, n_layers=5).to(device)
        self.disLB = Discriminator(input_nc=3, ndf=ch, n_layers=5).to(device)

        # ======================================================
        # Rho Clipper (for AdaILN & ILN)
        # ======================================================
        self.Rho_clipper = RhoClipper(0, 1)

        # =====================================================
        # Losses
        # =====================================================
        self.L1_loss = nn.L1Loss().to(cfg.device)
        self.MSE_loss = nn.MSELoss().to(cfg.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(cfg.device)

    def build_optimizers_and_schedulers(self):
        """
        Create optimizers and schedulers for generator and discriminator.
        Returns:
            (G_optim, D_optim, G_scheduler, D_scheduler)
        """

        # --- Optimizers ---
        G_optim = torch.optim.Adam(
            itertools.chain(self.genA2B.parameters(),
                            self.genB2A.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.wd
        )

        D_optim = torch.optim.Adam(
            itertools.chain(
                self.disGA.parameters(), self.disGB.parameters(),
                self.disLA.parameters(), self.disLB.parameters()
            ),
            lr=self.lr * 0.5,      # smaller LR for discriminators
            betas=(0.5, 0.999),
            weight_decay=self.wd
        )

        # --- Schedulers (linear decay after decay_start_epoch) ---
        def lambda_rule(epoch):
            if epoch < self.decay_start_epoch:
                return 1.0
            else:
                return 1.0 - (epoch - self.decay_start_epoch) / float(self.total_epochs - self.decay_start_epoch)

        G_scheduler = torch.optim.lr_scheduler.LambdaLR(
            G_optim, lr_lambda=lambda_rule)
        D_scheduler = torch.optim.lr_scheduler.LambdaLR(
            D_optim, lr_lambda=lambda_rule)

        return G_optim, D_optim, G_scheduler, D_scheduler

    def forward(self, real_A, real_B):

        ################### Discriminator ########################
        fake_A2B, _, _ = self.genA2B(real_A)
        fake_B2A, _, _ = self.genB2A(real_B)

        disc_real_logits = {
            'GA': self.disGA(real_A),
            'LA': self.disLA(real_A),
            'GB': self.disGB(real_B),
            'LB': self.disLB(real_B)
        }
        disc_fake_logits = {
            'GA': self.disGA(fake_B2A),
            'LA': self.disLA(fake_B2A),
            'GB': self.disGB(fake_A2B),
            'LB': self.disLB(fake_A2B)
        }

        D_loss = 0
        for key in ['GA', 'LA', 'GB', 'LB']:
            D_loss += self.adv_weight * \
                adversarial_loss(disc_real_logits[key][0],
                                 disc_fake_logits[key][0], self.MSE_loss)
            D_loss += self.adv_weight * \
                adversarial_loss(disc_real_logits[key][1],
                                 disc_fake_logits[key][1], self.MSE_loss)

        ################# Generator ##############################
        fake_A2B, fake_A2B_cam, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam, _ = self.genB2A(real_B)

        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam, _ = self.genA2B(real_B)

        # Discriminator outputs for fake images
        gen_logits = {
            'GA': self.disGA(fake_B2A),
            'LA': self.disLA(fake_B2A),
            'GB': self.disGB(fake_A2B),
            'LB': self.disLB(fake_A2B)
        }

        G_adv_loss = 0
        for key in ['GA', 'LA', 'GB', 'LB']:
            G_adv_loss += self.adv_weight * \
                generator_adv_loss(gen_logits[key][0], self.MSE_loss)
            G_adv_loss += self.adv_weight * \
                generator_adv_loss(gen_logits[key][1], self.MSE_loss)

        # Cycle consistency
        G_cycle_loss = self.cycle_weight * (cycle_loss(fake_A2B2A, real_A, self.L1_loss) +
                                            cycle_loss(fake_B2A2B, real_B, self.L1_loss))

        # Identity loss
        G_identity_loss = self.identity_weight * (identity_loss(fake_A2A, real_A, self.L1_loss) +
                                                  identity_loss(fake_B2B, real_B, self.L1_loss))

        # CAM loss
        G_cam_loss = self.cam_weight * (cam_loss(fake_B2A_cam, fake_A2A_cam, self.BCE_loss) +
                                        cam_loss(fake_A2B_cam, fake_B2B_cam, self.BCE_loss))

        G_loss = G_adv_loss + G_cycle_loss + G_identity_loss + G_cam_loss

        return D_loss, G_loss
