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
    or globals().get("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
)
PROMPT_FILE = (
    os.environ.get("PROMPT_FILE")
    or globals().get("PROMPT_FILE", "data/renellm/jb400.csv")
)
OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR")
    or globals().get("OUTPUT_DIR", "output/scan_heads_only/mistral7b")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

AFFIRM_TOKENS = (
    os.environ.get("AFFIRM_TOKENS")
    or globals().get("AFFIRM_TOKENS", "Sure,sure,Yes,yes,Alright,alright,The,Here,here")
)
REFUSE_TOKENS = (
    os.environ.get("REFUSE_TOKENS")
    or globals().get("REFUSE_TOKENS", "Sorry,sorry,Cannot,cannot,No,no,not,I'm,can't")
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
    or globals().get("LOG_FILE", "logs/mistral_scan_heads.log")
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

logger = logging.getLogger("ScanHeads")

##############################################################################
# 2. Load data line-by-line
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
if isinstance(NUM_SAMPLES, str) and NUM_SAMPLES.isdigit():
    NUM_SAMPLES = int(NUM_SAMPLES)
if NUM_SAMPLES:
    prompts = prompts[:NUM_SAMPLES]

logger.info(f"Loaded {len(prompts)} prompts from {PROMPT_FILE}.")

##############################################################################
# 3. Load model
##############################################################################
torch.cuda.empty_cache()
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available.")

logger.info(f"Loading tokenizer for {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading model for {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    use_auth_token=HF_TOKEN
)
model.eval()
logger.info("Model loaded successfully.")

##############################################################################
# 4. Prepare for logit difference
##############################################################################
def parse_token_list(toks_str, tokenizer):
    items = [x.strip() for x in toks_str.split(",")]
    valid = []
    for it in items:
        out = tokenizer(it, add_special_tokens=False)
        # only keep single-subword tokens
        if len(out["input_ids"]) == 1:
            valid.append(out["input_ids"][0])
    return list(set(valid))

affirm_ids = parse_token_list(AFFIRM_TOKENS, tokenizer)
refuse_ids = parse_token_list(REFUSE_TOKENS, tokenizer)
logger.info(f"AFFIRM tokens => {affirm_ids}")
logger.info(f"REFUSE tokens => {refuse_ids}")

def compute_logit_diff(logits_1d: torch.Tensor):
    """
    sum(refuse) - sum(affirm)
    """
    sum_ref = sum(logits_1d[x].item() for x in refuse_ids if 0 <= x < logits_1d.shape[0])
    sum_aff = sum(logits_1d[x].item() for x in affirm_ids if 0 <= x < logits_1d.shape[0])
    return sum_ref - sum_aff

##############################################################################
# 5. We'll define a function that ablates a single head in a single layer
##############################################################################
def make_head_hook(head_idx, start, end):
    """
    Return a forward hook that zeroes out hidden_states[..., start:end].
    This will be attached only to the single layer we want to ablate.
    """
    def hook_fn(module, inp, out):
        # out might be (hidden_states, attn_weights)
        if isinstance(out, tuple) and len(out) == 2:
            hidden_states, attn_weights = out
            hidden_states_mod = hidden_states.clone()
            hidden_states_mod[..., start:end] = 0.0
            return (hidden_states_mod, attn_weights)
        else:
            # single tensor
            out_clone = out.clone()
            out_clone[..., start:end] = 0.0
            return out_clone
    return hook_fn

##############################################################################
# 6. We'll define a function to measure logit diff with a given (layer,head) ablation
##############################################################################
@torch.no_grad()
def get_logit_diff(prompt, layer_idx=None, head_idx=None):
    """
    If layer_idx and head_idx are given, we zero out that single head in that single layer.
    Otherwise, no ablation (baseline).
    """
    # We'll attach the forward hook ONLY to the self_attn of the chosen layer
    handle = None
    if layer_idx is not None and head_idx is not None:
        if 0 <= layer_idx < len(model.model.layers):
            # get the submodule
            attn_module = model.model.layers[layer_idx].self_attn
            hidden_dim = model.config.hidden_size
            num_heads = model.config.num_attention_heads
            head_dim = hidden_dim // num_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            # build hook
            hook_fn = make_head_hook(head_idx, start, end)
            handle = attn_module.register_forward_hook(hook_fn)

    # run forward
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = enc["input_ids"].cuda()
    attn_mask = enc["attention_mask"].cuda()

    out = model(input_ids=input_ids, attention_mask=attn_mask)

    # remove hook if any
    if handle is not None:
        handle.remove()

    logits = out.logits  # [batch, seq_len, vocab_size]
    final_logits = logits[0, -1, :]  # next-token distribution
    return compute_logit_diff(final_logits)

##############################################################################
# 7. Baseline
##############################################################################
# measure the logit diff on each prompt with no ablation
baseline_diffs = [get_logit_diff(p, layer_idx=None, head_idx=None) for p in prompts]
avg_baseline = sum(baseline_diffs)/len(baseline_diffs) if baseline_diffs else 0.0
logger.info(f"Average baseline logit diff over {len(prompts)} prompts: {avg_baseline}")

##############################################################################
# 8. Now loop over each layer and head
##############################################################################
num_layers = len(model.model.layers)
num_heads = model.config.num_attention_heads
hidden_dim = model.config.hidden_size
logger.info(f"num_layers={num_layers}, num_heads={num_heads}, hidden_dim={hidden_dim}")

results = []
for l_idx in range(num_layers):
    for h_idx in range(num_heads):
        # measure ablated diffs
        ablated = [get_logit_diff(p, layer_idx=l_idx, head_idx=h_idx) for p in prompts]
        avg_abl = sum(ablated)/len(ablated) if ablated else 0.0
        score = avg_abl - avg_baseline
        results.append({
            "type": "attn",
            "layer": l_idx,
            "head": h_idx,
            "mean_logit_diff_ablation": avg_abl,
            "importance_score": score
        })
        logger.debug(f"layer={l_idx}, head={h_idx}, mean_logit={avg_abl:.4f}, importance={score:.4f}")

df = pd.DataFrame(results)
out_csv = os.path.join(OUTPUT_DIR, "component_importance_heads_only.csv")
df.to_csv(out_csv, index=False)
logger.info(f"Wrote heads-only results to: {out_csv}")
logger.info("=== Done scanning heads ===")
