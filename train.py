import torch
import torch.nn as nn
from core.UGATIT import UGATIT
from core.dataset import fetch_dataloader
import os
from tools import log_model_details, TeeOutput, cfg_to_dict
import datetime
import sys
from tqdm import tqdm

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        if torch.backends.cudnn.enabled and self.cfg.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print("#" * 64)
        print(cfg)
        print('#' * 64)

    def train(self):

        model = nn.DataParallel(UGATIT(self.cfg))
        model.build_model()

        model.cuda()
        model.train()

        Gen_optimizer, Disc_optimizer, Gen_scheduler, Disc_scheduler = model.build_optimizers_and_schedulers()

        train_loader = fetch_dataloader(self.cfg)
        should_keep_training = True

        log_file_path = os.path.join("checkpoints/", "training_log.txt")
        model_file_path = os.path.join("checkpoints/", "info.txt")

        file_mode = 'a' if self.cfg.resume else 'w'
        with open(log_file_path, file_mode, encoding='utf-8') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"{'RESUMED' if self.cfg.resume else 'NEW'} TRAINING SESSION: {
                    datetime.datetime.now()}\n")
            f.write("Command line: " + " ".join(sys.argv) + "\n")
            f.write("Configuration:\n")
            f.write(str(cfg_to_dict(self.cfg)))
            f.write("\n" + "="*50 + "\n")

        tee = TeeOutput(log_file_path)
        original_stdout = sys.stdout
        sys.stdout = tee

        def save_checkpoint(step, path=None):
            ckpt = {
                'step': step,
                'model': model.state_dict(),
                'disc_optimizer': Disc_optimizer.state_dict(),
                'disc_scheduler': Disc_scheduler.state_dict(),
                'gen_optimizer': Gen_optimizer.state_dict(),
                'gen_scheduler': Gen_scheduler.state_dict()
            }
            if not path:
                path = os.path.join("checkpoints/train/",
                                    f"resume_ckpt_{step}.pth")
            torch.save(ckpt, path)
            print(f"💾 Saved checkpoint at step {step}: {path}")

        if getattr(self.cfg, 'resume', False) and self.cfg.resume_ckpt_path:
            ckpt = torch.load(self.cfg.resume_ckpt_path)
            model.load_state_dict(ckpt['model'])
            total_steps = ckpt['step']
            Disc_optimizer.load_state_dict(ckpt['disc_optimizer'])
            Disc_scheduler.load_state_dict(ckpt['disc_scheduler'])
            Gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
            Gen_scheduler.load_state_dict(ckpt['gen_scheduler'])
            print(f"🔄 Resumed from step {total_steps}")
        else:
            total_steps = 0
            print(f"🆕 Training from scratch")

        while should_keep_training:
            pbar = tqdm(enumerate(train_loader),
                        total=self.cfg.iterations, initial=total_steps)
            for idx, batch in enumerate(train_loader):
                Gen_optimizer.zero_grad()
                Disc_optimizer.zero_grad()
                real_A = batch['real_A'].cuda()
                real_B = batch['real_B'].cuda()

                # Forward Pass
                discriminator_loss, generator_loss = model(real_A, real_B)

                discriminator_loss.backward()
                generator_loss.backward()

                Disc_optimizer.step()
                Disc_scheduler.step()
                Gen_optimizer.step()
                Gen_scheduler.step()
                total_steps += 1

                if total_steps % cfg.val_freq == cfg.val_freq - 1:
                    save_checkpoint(total_steps+1)

                if (total_steps  > self.cfg.iterations):
                    should_keep_training = False
                    break


        save_checkpoint(total_steps, "checkpoints/train/final.pth")
        torch.save(model.state_dict(), PATH)
