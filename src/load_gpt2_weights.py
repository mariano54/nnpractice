import dataclasses
from typing import List

from transformers import GPT2LMHeadModel
import pickle
import torch

from src.gpt2_weights import GPT2Weights, TransformerWeights, AttentionWeights, MLPWeights


# Use this to generate a "gpt2_weights.pkl" file that is used by our neural net
def weights_from_transformers() -> GPT2Weights:
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    for k, v in model_hf.items():
        print(k, v.shape)

    wte = model_hf["transformer.wte.weight"]
    wpe = model_hf["transformer.wpe.weight"]
    transformer_ln_f_weight = model_hf["transformer.ln_f.weight"]
    transformer_ln_f_bias = model_hf["transformer.ln_f.bias"]
    lm_head_weight = model_hf["lm_head.weight"]
    p_names_1= ['ln_1.weight', 'ln_1.bias']
    p_names_2 = ['attn.c_proj.weight', 'attn.c_proj.bias',
               'ln_2.weight', 'ln_2.bias', 'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias']

    # transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    transformers: List[TransformerWeights] = []
    for i in range(12):
        # Splits up q,k,v since it's easier to understand when they are split
        param_strs_1 = [f"transformer.h.{i}.{p_name}" for p_name in p_names_1]
        param_strs_2 = [f"transformer.h.{i}.{p_name}" for p_name in p_names_2]
        q_weight, k_weight, v_weight = model_hf[f"transformer.h.{i}.attn.c_attn.weight"].split(768, dim=1)
        q_bias, k_bias, v_bias  = model_hf[f"transformer.h.{i}.attn.c_attn.bias"].split(768, dim=0)
        ws_1 = [model_hf[p_str] for p_str in param_strs_1]
        ws_2 = [model_hf[p_str] for p_str in param_strs_2]
        att = AttentionWeights(q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, ws_2[0], ws_2[1])
        mlp = MLPWeights(ws_2[4], ws_2[5], ws_2[6], ws_2[7])

        transformers.append(TransformerWeights(*(ws_1 + [att, mlp] + ws_2[2:4])))
    return GPT2Weights(wte, wpe, transformers, transformer_ln_f_weight, transformer_ln_f_bias, lm_head_weight)


w = weights_from_transformers()
pickle.dump(w, open("data/gpt2_weights.pkl", "wb"))
w2: GPT2Weights = pickle.load(open("data/gpt2_weights.pkl", "rb"))

print(torch.allclose(w.transformer[3].ln_1_weight, w2.transformer[3].ln_1_weight))


