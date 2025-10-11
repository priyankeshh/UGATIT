from yacs.config import CfgNode as CN

_CN = CN()

# ======================================================
# Basic Experiment Settings
# ======================================================
_CN.phase = "train"                     # [train / test]
_CN.light = True                       # U-GAT-IT light version if True
_CN.dataset = "YOUR_DATASET_NAME"       # dataset name
_CN.result_dir = "results"              # directory to save results
_CN.device = "cuda"                     # [cpu / cuda]
_CN.benchmark_flag = False
_CN.resume = False
_CN.resume_ckpt_path = ""

# ======================================================
# Training Parameters
# ======================================================
_CN.iterations = 1000000                 # total training iterations
_CN.batch_size = 1                      # batch size
_CN.print_freq = 100                   # print image frequency
_CN.save_freq = 100000                  # model save frequency
_CN.val_freq = 10000
_CN.decay_flag = True                   # enable learning rate decay
_CN.num_workers = 4

# ======================================================
# Optimization Parameters
# ======================================================
_CN.lr = 0.0001                         # learning rate
_CN.weight_decay = 0.0001               # weight decay (L2 regularization)
_CN.adv_weight = 1                      # weight for GAN
_CN.cycle_weight = 10                   # weight for cycle-consistency loss
_CN.identity_weight = 10                # weight for identity loss
_CN.cam_weight = 1000                   # weight for CAM loss

# ======================================================
# Architecture Parameters
# ======================================================
_CN.ch = 64                             # base channel number per layer
_CN.n_res = 4                           # number of residual blocks
_CN.n_dis = 6                           # number of discriminator layers

# ======================================================
# Image Specifications
# ======================================================
_CN.img_size = 256                      # image size
_CN.img_ch = 3                          # number of channels in the image

# ======================================================
# Model Section
# ======================================================
_CN.UGATIT = CN()
_CN.UGATIT.name = "UGATIT"
_CN.UGATIT.version = "light"            # or "full"
_CN.UGATIT.critical_params = [
    "light", "ch", "n_res", "n_dis", "img_size", "img_ch"
]

# ======================================================
# Trainer Section
# ======================================================
_CN.trainer = CN()
_CN.trainer.scheduler = "linear"        # can be OneCycleLR, cosine, etc.
_CN.trainer.optimizer = "adam"          # [adam / adamw / sgd]
_CN.trainer.beta1 = 0.5
_CN.trainer.beta2 = 0.999
_CN.trainer.epsilon = 1e-8
_CN.trainer.clip_grad = 1.0             # gradient clipping
_CN.trainer.num_steps = 1000000
_CN.trainer.warmup_steps = 0
_CN.trainer.decay_strategy = "linear"

# ======================================================
# Function to get config clone
# ======================================================


def get_cfg():
    """Return a clone of the default UGATIT configuration."""
    return _CN.clone()
