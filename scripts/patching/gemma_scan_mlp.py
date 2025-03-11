import os
import sys
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
# 1. Config
##############################################################################
MODEL_NAME = (
    os.environ.get("MODEL_NAME")
    #or globals().get("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
    or globals().get("MODEL_NAME", "google/gemma-2-2b-it")
)
PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE", "data/renellm/jb400.csv")
)
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR", "output/scan_mlp/gemma2b")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

AFFIRM_TOKENS = (
    os.environ.get("AFFIRM_TOKENS")
    or globals().get("AFFIRM_TOKENS", "Sure,Yes,Alright,The,Here")
)
REFUSE_TOKENS = (
    os.environ.get("REFUSE_TOKENS")
    or globals().get("REFUSE_TOKENS", "Sorry,Cannot,No,I'm,can't")
)

MAX_SEQ_LENGTH = int(
    os.environ.get("MAX_SEQ_LENGTH")
    or globals().get("MAX_SEQ_LENGTH", 512)
)
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or globals().get("HF_TOKEN", None)
)

LOG_FILE = (
    os.environ.get("LOG_FILE")
    or globals().get("LOG_FILE", "logs/gemma_scan_mlp.log")
)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logger = logging.getLogger("ScanMLP")

##############################################################################
# 2. Load data
##############################################################################
def load_prompts(file_path):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                lines.append(line)
    return lines

prompts = load_prompts(PROMPT_FILE)
NUM_SAMPLES = (
    os.environ.get("NUM_SAMPLES")
    or globals().get("NUM_SAMPLES", None)
)
NUM_SAMPLES = None
if NUM_SAMPLES and NUM_SAMPLES.isdigit():
    n=int(NUM_SAMPLES)
    prompts = prompts[:n]

logger.info(f"Loaded {len(prompts)} lines from {PROMPT_FILE}")

##############################################################################
# 3. Load model
##############################################################################
torch.cuda.empty_cache()
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available")

logger.info(f"Loading tokenizer from {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading model from {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    use_auth_token=HF_TOKEN
)
model.eval()
logger.info("Model loaded OK")

##############################################################################
# 4. Parse tokens
##############################################################################
def parse_token_list(toks, tokenizer):
    items = [x.strip() for x in toks.split(",")]
    valid = []
    for it in items:
        out = tokenizer(it, add_special_tokens=False)
        if len(out["input_ids"]) == 1:
            valid.append(out["input_ids"][0])
    return list(set(valid))

affirm_ids = parse_token_list(AFFIRM_TOKENS, tokenizer)
refuse_ids = parse_token_list(REFUSE_TOKENS, tokenizer)

def compute_logit_diff(logits_1d):
    sum_ref = sum(logits_1d[x].item() for x in refuse_ids if x<logits_1d.shape[0])
    sum_aff = sum(logits_1d[x].item() for x in affirm_ids if x<logits_1d.shape[0])
    return sum_ref - sum_aff

@torch.no_grad()
def get_logit_diff(prompt, hook=None):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids=enc["input_ids"].cuda()
    attention_mask=enc["attention_mask"].cuda()

    h=[]
    if hook is not None:
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                h.append(layer.mlp.register_forward_hook(hook))

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    for x in h:
        x.remove()

    logits = out.logits
    final_logits = logits[0, -1, :]
    return compute_logit_diff(final_logits)

##############################################################################
# 5. We only do MLP scanning
##############################################################################
num_layers = len(model.model.layers)
logger.info(f"num_layers={num_layers}")

##############################################################################
# 5b. Baseline diffs
##############################################################################
baseline = [get_logit_diff(p) for p in prompts]
avg_base = sum(baseline)/len(baseline)

logger.info(f"Average baseline: {avg_base}")

##############################################################################
# 6. MLP hooking
##############################################################################
def mlp_hook_factory():
    """Zero entire MLP output."""
    def hook_fn(module, inp, out):
        # out shape: [batch, seq, hidden_dim]
        return torch.zeros_like(out)
    return hook_fn

##############################################################################
# 7. For each layer, we zero out MLP output
##############################################################################
results = []
for layer_idx in range(num_layers):
    def layer_hook(module, inp, out):
        # This ensures we only ablate the MLP in the chosen layer
        # If "module" is the MLP in layer layer_idx
        return torch.zeros_like(out)

    # But we must attach specifically to the MLP in this layer
    # Easiest approach is per-prompt hooking:
    diffs=[]
    for p in prompts:
        # We'll do a forward hook *just for that one layer*
        handle = model.model.layers[layer_idx].mlp.register_forward_hook(layer_hook)
        d = get_logit_diff(p, hook=None)  # "hook=None" because we manually attached above
        diffs.append(d)
        handle.remove()

    mean_diff = sum(diffs)/len(diffs)
    score = mean_diff - avg_base
    results.append({
        "type": "mlp",
        "layer": layer_idx,
        "mean_logit_diff_ablation": mean_diff,
        "importance_score": score
    })
    logger.debug(f"layer={layer_idx}, ablation_score={score}")

df = pd.DataFrame(results)
out_csv = os.path.join(OUTPUT_DIR, "component_importance_mlp_only.csv")
df.to_csv(out_csv, index=False)
logger.info(f"Wrote MLP-only results to {out_csv}")
logger.info("=== Done scanning MLPs ===")
