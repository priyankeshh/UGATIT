import sys
from torchinfo import summary
import hashlib
import requests


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TeeOutput:
    """Class to duplicate output to both terminal and log file."""

    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def cfg_to_dict(cfg):
    """Recursively convert a YACS CfgNode into a pure Python dictionary."""
    if not hasattr(cfg, "items"):
        return cfg
    out = {}
    for k, v in cfg.items():
        out[k] = cfg_to_dict(v)
    return out


def log_model_details(model, log_file_path, config_dict=None, input_size=None):
    """
    Log detailed model architecture using torchinfo.summary along with parameter counts.
    """
    try:
        if input_size is None:
            input_size = ((1, 3, 256, 256), (1, 3, 256, 256))

        # Get the detailed model summary using torchinfo
        model_summary = summary(model, input_size=input_size, verbose=0)
        summary_str = str(model_summary)
        # print(f"📋 Model Architecture Summary:")
        # print(summary_str)
    except Exception as e:
        print(f"Warning: torchinfo.summary failed with error:"
              f"{e}. Using basic model string.")
        summary_str = str(model)
        # print(f"📋 Model Structure:")
        # print(summary_str)

    # Compute parameter counts
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel()
                          for p in model.parameters() if p.requires_grad)

    print(f"📊 Parameter Details:")
    print(f"  Total Parameters: {param_count:,}")
    print(f"  Trainable Parameters: {trainable_count:,}")
    print(f"  Non-trainable Parameters: {param_count - trainable_count:,}")

    full_summary = (
        f"Model Detailed Summary:\n{'-'*50}\n"
        f"{summary_str}\n\n"
        f"Total Parameters: {param_count:,}\n"
        f"Trainable Parameters: {trainable_count:,}\n"
        f"Non-trainable Parameters: {param_count - trainable_count:,}\n"
    )

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(full_summary)
        if config_dict:
            f.write("\nTraining Configuration:\n")
            f.write(f"{'-'*50}\n")
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

    return full_summary


def is_online() -> bool:
    """Check for internet connectivity by pinging Google."""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def generate_run_id(model_name, session_name):
    """Generate a consistent run ID based on model name and session."""
    # Create a hash from model name and session for consistency
    run_string = f"{model_name}_{session_name}"
    run_hash = hashlib.md5(run_string.encode()).hexdigest()[:8]
    return f"{model_name}_{session_name}_{run_hash}"
