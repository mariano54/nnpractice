from contextlib import nullcontext

import torch

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
torch.set_float32_matmul_precision("high")
device = "cuda"
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
print(f"PTDtype: {ptdtype}")
ctx = (
    nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

class ConditionalAutocast:
    def __init__(self, use_autocast):
        self.use_autocast = use_autocast
        self.autocast_context = None

    def __enter__(self):
        if self.use_autocast:
            self.autocast_context = ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.autocast_context:
            self.autocast_context.__exit__(exc_type, exc_val, exc_tb)