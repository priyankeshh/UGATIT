import torch
import torch.nn as nn
from core.UGATIT import UGATIT
from core.dataset import fetch_dataloader
import os
from tools import log_model_details, TeeOutput, cfg_to_dict, count_parameters, is_online, generate_run_id
from loguru import logger as loguru_logger
from core.utils import Logger
import datetime
import sys
from tqdm import tqdm
import wandb
from evaluate import validate


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

        model = nn.DataParallel(UGATIT())
        model.module.build_model(self.cfg)

        model.cuda()
        model.train()

        Gen_optimizer, Disc_optimizer, Gen_scheduler, Disc_scheduler = model.module.build_optimizers_and_schedulers()

        loguru_logger.info("Parameter Count: %d" % count_parameters(model))

        train_loader, val_loader = fetch_dataloader(self.cfg)
        should_keep_training = True

        base_path = "/kaggle/working/"
        train_path = os.path.join(base_path, "train/")
        os.makedirs(train_path, exist_ok=True)

        log_file_path = os.path.join(base_path, "training_log.txt")
        model_file_path = os.path.join(base_path, "info.txt")
        log_model_details(model, model_file_path, self.cfg,
                          ((1, 3, 256, 256), (1, 3, 256, 256)))

        file_mode = 'a' if self.cfg.resume else 'w'
        with open(log_file_path, file_mode, encoding='utf-8') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"{'RESUMED' if self.cfg.resume else 'NEW'}"
                    f"TRAINING SESSION: {datetime.datetime.now()}\n")
            f.write("Command line: " + " ".join(sys.argv) + "\n")
            f.write("Configuration:\n")
            f.write(str(cfg_to_dict(self.cfg)))
            f.write("\n" + "="*50 + "\n")

        tee = TeeOutput(log_file_path)
        original_stdout = sys.stdout
        sys.stdout = tee

        logger = Logger(model, self.cfg)

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
                path = os.path.join(train_path,
                                    f"resume_ckpt_{step}.pth")
            torch.save(ckpt, path)
            artifact = wandb.Artifact(name="generator_ckpt", type="model")
            artifact.add_file(local_path=path)
            artifact.save()
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

        wandb_enabled = True
        wandb_run = None
        if wandb_enabled:
            try:
                api_key = os.environ['WANDB_KEY']
                wandb.login(key=api_key)
                run_name = f"image_translation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

                if is_online():
                    wandb_run = wandb.init(
                        project="UGATIT",
                        config=cfg_to_dict(self.cfg),
                        name=run_name,
                        id=generate_run_id("Selfie2Anime", "_train2"),
                        resume="allow"
                    )
                else:
                    wandb_run = wandb.init(
                        project="UGATIT",
                        config=cfg_to_dict(self.cfg),
                        name=run_name,
                        id=generate_run_id("Selfie2Anime", "_train2"),
                        mode='offline'
                    )
                print("✅ WandB initialized for stage_1_train")
            except Exception as e:
                print(f"❌ WandB init failed: {e}")
                wandb_enabled = False

        while should_keep_training:
            pbar = tqdm(enumerate(train_loader),
                        total=self.cfg.iterations, initial=total_steps)
            for idx, batch in pbar:
                Gen_optimizer.zero_grad()
                Disc_optimizer.zero_grad()
                real_A = batch['real_A'].cuda()
                real_B = batch['real_B'].cuda()

               # Forward Pass
                discriminator_loss, generator_loss, metrics = model(
                    real_A, real_B)

                discriminator_loss.mean().backward(retain_graph=True)
                generator_loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.cfg.trainer.clip_grad)
                model.module.genA2B.apply(model.module.Rho_clipper)
                model.module.genB2A.apply(model.module.Rho_clipper)

                Disc_optimizer.step()
                Disc_scheduler.step()
                Gen_optimizer.step()
                Gen_scheduler.step()
                total_steps += 1

                metrics = {k: v.mean().item() for k, v in metrics.items()}

                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

                logger.push(metrics)

                if wandb_enabled and (total_steps % 100 == 0):
                    wandb_metrics = {
                        "steps": total_steps,
                        "total_loss": discriminator_loss.mean().item() + generator_loss.mean().item()
                    }

                    train_metrics = dict(metrics, **wandb_metrics)
                    wandb_run.log(train_metrics)

                if total_steps % self.cfg.val_freq == self.cfg.val_freq - 1:
                    save_checkpoint(total_steps+1)

                    val_metrics = validate(model, val_loader, total_steps)
                    if wandb_enabled:
                        try:
                            wandb_run.log({
                                "Validation/metrics": {k: v for k, v in val_metrics.items() if k != 'images'},
                                "Validation/RealA_FakeB": [wandb.Image(val_metrics['images']['RealA_FakeB'], caption="RealA → FakeB")],
                                "Validation/RealB_FakeA": [wandb.Image(val_metrics['images']['RealB_FakeA'], caption="RealB → FakeA")]
                            })
                        except Exception as e:
                            print(f"Error logging to WandB: {e}")
                    model.train()

                if (total_steps > self.cfg.iterations):
                    should_keep_training = False
                    break

        final_ckpt = os.path.join(base_path, "final_ckpt.pth")
        final_model = os.path.join(base_path, "model_final.pth")
        save_checkpoint(total_steps, final_ckpt)
        torch.save(model.state_dict(), final_model)
        artifact = wandb.Artifact(name="generator_ckpt", type="model")
        artifact.add_file(local_path=final_model)
        artifact.save()

        wandb_run.finish()

    
