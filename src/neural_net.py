from typing import List, Optional, Tuple

import math
import torch

from src.gpt2_weights import GPT2Weights, TransformerWeights, AttentionWeights
from src.random_weights import linear_rw, layer_batch_norm_rw

device = "cuda"
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
print(f"Using {device} device")


class GPT2Model:
    def __init__(
        self,
        weights: Optional[GPT2Weights],
        B: int,
        T: int,
        C: int,
        vocab_size: int,
        n_heads: int,
        dropout: float,
    ):
        self.dtoken_embeddings = None
        self.dpos_embeddings = None
        self.max_B = B
        self.max_T = T
        self.C = C
        self.B = None
        self.T = None
        self.xs = None
        self.vocab_size = vocab_size
        n_layers = len(weights.transformer)
        assert int(C / n_heads) == C / n_heads

        self.initial_dropout = Dropout(dropout)
        if weights is None:
            std = 1 / math.sqrt(C)
            self.token_embeddings = torch.normal(0, std, (vocab_size, C)).to(device)
            self.positional_embeddings = torch.normal(0, std, (T, C)).to(device)
            self.transformer_blocks = [
                TransformerBlock(
                    n_embed=C,
                    n_heads=n_heads,
                    dropout=dropout,
                    transformer_weights=None,
                )
                for _ in range(n_layers)
            ]
            self.final_ln = LayerNorm(None, None, C)
            self.lm_head = self.token_embeddings
        else:
            self.token_embeddings = weights.wte
            self.positional_embeddings = weights.wpe
            self.transformer_blocks = [
                TransformerBlock(
                    n_embed=C,
                    n_heads=n_heads,
                    dropout=dropout,
                    transformer_weights=weights.transformer[i],
                )
                for i in range(n_layers)
            ]
            self.final_ln = LayerNorm(
                weights.transformer_ln_f_weight, weights.transformer_ln_f_bias, None
            )
            self.pre_lm_head = None
            self.lm_head = weights.wte

        self.final_softmax = Softmax()

    def forward(
        self,
        xs: torch.Tensor,
        ys: Optional[torch.Tensor],
        topk: Optional[int] = None,
        temperature: float = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # xs and ys are shape (B,T)
        assert xs.shape[0] <= self.max_B
        assert xs.shape[1] <= self.max_T
        self.xs = xs.clone()

        # TODO: enable inputs of different sizes within batch

        embeddings = self.token_embeddings[xs]  # Should result in B,T,C vector
        position_emb = self.positional_embeddings[
            torch.arange(0, xs.shape[1]).to(device)
        ]

        start = embeddings + position_emb
        self.B, self.T, C = start.shape

        filtered_start = self.initial_dropout.forward(start, True)
        inter0 = filtered_start
        self.all_transformer_inters = []
        for block in self.transformer_blocks:
            inter0 = block.forward(inter0, True)
            self.all_transformer_inters.append(inter0)
        self.inter0 = inter0
        self.inter = self.final_ln.forward(inter0, True)

        if ys is not None:
            self.pre_lm_head = self.inter.view(
                (self.B * self.T), C
            )  # B*T training examples
            self.logits = self.pre_lm_head @ self.lm_head.T
            self.final_softmax.forward(self.logits, ys is not None)
            loss = -torch.mean(
                torch.log(
                    self.final_softmax.probs[
                        range(self.final_softmax.probs.shape[0]),
                        ys.view((self.B * self.T)),
                    ]
                )
            )
            print(f"Loss: {loss}")
            return self.final_softmax.probs.view((self.B, self.T, self.vocab_size)), loss
        else:
            assert topk is not None
            logits = (self.inter[:, -1, :] @ self.lm_head.T) / temperature
            vals, _ = torch.topk(logits, topk, 1)
            logits[logits < vals[:, [-1]]] = float("-inf")
            self.final_softmax.forward(logits, ys is not None)
            return self.final_softmax.probs, None

    def generate(
        self, xs: torch.Tensor, n_tokens: int, topk: int, temperature: float = 1.0
    ):
        B = xs.shape[0]  # batch size
        initial_len = xs.shape[1]
        assert B <= self.max_B
        while xs.shape[1] < initial_len + n_tokens:
            last_word_probs, _ = self.forward(
                xs[:, -self.max_T :], None, topk, temperature
            )
            last_gen_token: List[int] = []
            for batch_index in range(B):
                one_stream_probs = last_word_probs[batch_index]
                sampled_token = torch.multinomial(one_stream_probs, num_samples=1)
                last_gen_token.append(sampled_token.item())
            xs = torch.cat(
                (xs, torch.tensor(last_gen_token).unsqueeze(1).to(device)), dim=1
            )
        return xs

    def backward(self, labels: torch.Tensor) -> None:
        assert self.final_softmax.probs is not None
        dlogits = self.final_softmax.backward_cross_entropy(
            labels.view((self.B * self.T), 1)
        )

        dpre_lm_head = dlogits @ self.lm_head
        dlm_head_w = self.pre_lm_head.T @ dlogits

        doutput = dpre_lm_head.view(self.B, self.T, self.C)
        doutput = self.final_ln.backward(doutput)

        for block in reversed(self.transformer_blocks):
            doutput = block.backward(doutput)

        doutput = self.initial_dropout.backward(doutput)

        self.dtoken_embeddings = torch.zeros_like(self.token_embeddings)
        self.dpos_embeddings = torch.zeros_like(self.positional_embeddings)
        for i in range(self.xs.shape[0]):
            for j in range(self.xs.shape[1]):
                self.dtoken_embeddings[self.xs[i, j]] += doutput[i, j]
                self.dpos_embeddings[j] += doutput[i, j]
        self.dtoken_embeddings += dlm_head_w.T

    def apply_gradient(self, learning_rate: float) -> None:
        for layer in reversed(self.transformer_blocks):
            layer.apply_gradient(learning_rate)
        self.initial_dropout.apply_gradient(learning_rate)
        self.token_embeddings -= self.dtoken_embeddings * learning_rate
        self.positional_embeddings -= self.dpos_embeddings * learning_rate

    def get_weights(self) -> GPT2Weights:
        pass


class Relu:
    def __init__(self):
        self.last_preacts = None

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        self.last_preacts = x
        return torch.maximum(x, torch.zeros_like(x))

    def backward(self, doutput: torch.Tensor) -> torch.Tensor:
        d_activations = doutput
        dpreact_template = torch.zeros_like(doutput)
        over_zero = torch.nonzero(self.last_preacts > 0, as_tuple=False)
        dpreact_template[over_zero[:, 0], over_zero[:, 1]] = 1
        return dpreact_template * d_activations

    def apply_gradient(self, learning_rate: float):
        pass


class Gelu:
    def __init__(self):
        self.last_x = None

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        self.last_x = x.clone()
        last_phi_plus_1 = 1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
        last_phi_plus_1 *= 0.5 * x
        # del last_phi_plus_1
        # torch.cuda.empty_cache()
        return last_phi_plus_1

    def backward(self, doutput: torch.Tensor) -> torch.Tensor:
        last_phi = torch.tanh(
            math.sqrt(2 / math.pi) * (self.last_x + 0.044715 * self.last_x**3)
        )
        dgelu = 0.5 * (
            1
            + last_phi
            + self.last_x
            * math.sqrt(2 / math.pi)
            * (1 + 0.13145 * self.last_x**2)
            * (1 - last_phi**2)
        )
        return dgelu * doutput

    def apply_gradient(self, learning_rate: float):
        pass


class Softmax:
    def __init__(self, dimension: int = 1):
        self.probs = None
        self.dimension = dimension

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x2 = x - torch.max(x, dim=self.dimension, keepdim=True).values
        # exps = torch.exp(x)
        x3 = x2.exp()
        denominators = torch.sum(x3, dim=self.dimension, keepdim=True)

        # probs = (x / denominators)
        self.probs = x3.div(denominators)
        return self.probs

    def backward_cross_entropy(self, labels: torch.Tensor) -> torch.Tensor:
        n = self.probs.shape[0]
        d_logits = self.probs.clone()
        for i in range(n):
            d_logits[i][labels[i]] -= 1
        # d_logits[range(n), labels] -= 1
        return d_logits / n

    def backward(self, doutput: torch.Tensor, answer) -> torch.Tensor:
        doutput -= torch.sum(doutput * self.probs, dim=self.dimension, keepdim=True)
        dlogits = doutput * self.probs
        # probs_extended = self.probs.unsqueeze(2)
        # jacobian = torch.diagflat(self.probs) - torch.matmul(
        #     probs_extended, probs_extended.transpose(-2, -1)
        # )
        # dlogits = jacobian.t() @ doutput
        return dlogits

    def apply_gradient(self, learning_rate: float):
        pass


class Linear:
    def __init__(
        self,
        weights: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        compute_dx: bool = True,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        zero_biases: bool = False,
        residual_scaling_num_layers: Optional[int] = None,
    ):
        assert (weights is not None) == (bias is not None)
        if weights is not None:
            self.weights = weights
            self.bias = bias
        else:
            self.weights, self.bias = linear_rw(
                input_dim, output_dim, zero_biases, residual_scaling_num_layers, device
            )
        self.last_inputs = None
        self.dW = None
        self.dbias = None
        self.compute_dx = compute_dx

    def forward(self, xs: torch.Tensor, training: bool) -> torch.Tensor:
        self.last_inputs = xs.clone()
        ret = xs @ self.weights + self.bias
        return ret

    def forward_no_cache(self, xs: torch.Tensor) -> torch.Tensor:
        return xs @ self.weights + self.bias

    def backward(self, doutput: torch.Tensor) -> torch.Tensor:
        last_inputs_reshaped = self.last_inputs.reshape(
            -1, self.last_inputs.shape[-1]
        ).T
        self.dW = last_inputs_reshaped @ doutput.reshape(-1, doutput.shape[-1])
        self.dbias = doutput.sum((0, 1))
        if self.compute_dx:
            return doutput @ self.weights.T
        return torch.Tensor()

    def apply_gradient(self, learning_rate: float):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.dbias


class LayerNorm:
    def __init__(
        self,
        bn_gain: Optional[torch.Tensor],
        bn_bias: Optional[torch.Tensor],
        layer_size: Optional[int] = None,
    ):
        assert (bn_gain is None) == (bn_bias is None) == (layer_size is not None)
        if bn_bias is not None:
            self.bn_gain = bn_gain
            self.bn_bias = bn_bias
        else:
            self.bn_gain, self.bn_bias = layer_batch_norm_rw(layer_size, device)
        self.x_hat = None  # Used in backprop
        self.d_bias = None
        self.d_gain = None

    def forward(
        self,
        z: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        bn_mean = (1.0 / z.shape[2]) * (z.sum(2, keepdim=True))  # 32x1
        self.bn_diff = z - bn_mean  # B*T*C
        bn_diff_sq = self.bn_diff**2  # B*T*C
        self.bn_var = (1.0 / (z.shape[2])) * bn_diff_sq.sum(2, keepdim=True)  # 32x1

        bn_var_inv = (
            self.bn_var + 1e-5
        ) ** -0.5  # This is 1/sqrt(var + epsilon)  #32x1
        self.x_hat = self.bn_diff * bn_var_inv  # 32x1000
        preact = self.x_hat * self.bn_gain + self.bn_bias  # 32x1000
        return preact

    def backward(self, doutput: torch.Tensor):
        d_preact: torch.Tensor = doutput
        assert self.x_hat is not None
        layer_size = doutput.shape[2]

        bn_var_inv = (
            self.bn_var + 1e-5
        ) ** -0.5  # This is 1/sqrt(var + epsilon)  #32x1

        d_xhatdL = self.bn_gain * d_preact
        d_bnvarinvdL = (self.bn_diff * d_xhatdL).sum(2, keepdim=True)
        d_bnvardL = (-0.5 * (self.bn_var + 1e-5) ** (-1.5)) * d_bnvarinvdL
        d_bndiffsqrdL = (1.0 / layer_size) * d_bnvardL.expand(doutput.shape)
        d_bndiffdL = bn_var_inv * d_xhatdL + (2 * self.bn_diff) * d_bndiffsqrdL
        d_bn_meandL = (-1 * d_bndiffdL).sum(2, keepdim=True)
        d_z = d_bndiffdL + (1 / layer_size) * d_bn_meandL.expand(doutput.shape)

        self.d_gain = (self.x_hat * d_preact).sum((0, 1))
        self.d_bias = d_preact.sum((0, 1))
        # The following is taken from
        # github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynbo
        # dz = (
        #     self.bn_gain
        #     * self.bn_var_inv
        #     / n
        #     * (
        #         n * d_preact
        #         - d_preact.sum(0)
        #         # - n / (n - 1) * To align with pytorch, keep this commented
        #         - n/ (n-1)
        #         - self.x_hat * (d_preact * self.x_hat).sum(0)
        #     )
        # )

        return d_z

    def apply_gradient(self, learning_rate: float):
        self.bn_gain -= learning_rate * self.d_gain
        self.bn_bias -= learning_rate * self.d_bias


class BatchNorm:
    def __init__(
        self,
        bn_gain: Optional[torch.Tensor],
        bn_bias: Optional[torch.Tensor],
        momentum: float,
        layer_size: Optional[int] = None,
    ):
        assert (bn_gain is None) == (bn_bias is None) == (layer_size is not None)
        self.momentum = momentum
        if bn_bias is not None:
            self.bn_gain = bn_gain
            self.bn_bias = bn_bias
        else:
            self.bn_gain, self.bn_bias = layer_batch_norm_rw(layer_size, device)
        self.running_mean = torch.zeros(self.bn_bias.shape[0]).to(device)
        self.running_var = torch.ones(self.bn_bias.shape[0]).to(device)
        self.bn_var_inv = None  # Used in backprop
        self.x_hat = None  # Used in backprop
        self.d_bias = None
        self.d_gain = None

    def forward(
        self,
        z: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        if training:
            bn_mean = (1.0 / z.shape[0]) * (z.sum(0, keepdim=True))
            bn_diff = z - bn_mean
            bn_diff_sq = bn_diff**2
            bn_var = (1.0 / (z.shape[0])) * bn_diff_sq.sum(0, keepdim=True)

        else:
            bn_mean = self.running_mean
            bn_var = self.running_var
            bn_diff = z - bn_mean

        bn_var_inv = (bn_var + 1e-5) ** -0.5  # This is 1/sqrt(var + epsilon)
        x_hat = bn_diff * bn_var_inv
        preact = x_hat * self.bn_gain + self.bn_bias
        self.bn_var_inv = bn_var_inv
        self.x_hat = x_hat

        if training:
            self.running_mean = (1 - self.momentum) * self.running_mean + (
                self.momentum * bn_mean
            )
            self.running_var = (1 - self.momentum) * self.running_var + (
                self.momentum * bn_var
            )

        return preact

    def backward(self, doutput: torch.Tensor):
        d_preact: torch.Tensor = doutput
        assert self.bn_var_inv is not None
        assert self.x_hat is not None
        n = doutput.shape[0]
        # The following is taken from
        # github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipyn
        dz = (
            self.bn_gain
            * self.bn_var_inv
            / n
            * (
                n * d_preact
                - d_preact.sum(0)
                # - n / (n - 1) * To align with pytorch, keep this commented
                - self.x_hat * (d_preact * self.x_hat).sum(0)
            )
        )
        self.d_gain = (self.x_hat * d_preact).sum(0, keepdim=True)
        self.d_bias = d_preact.sum(0, keepdim=True)
        return dz

    def apply_gradient(self, learning_rate: float):
        self.bn_gain -= learning_rate * self.d_gain
        self.bn_bias -= learning_rate * self.d_bias


class Dropout:
    def __init__(self, dropout_prob: float):
        self.dropout_prob = dropout_prob
        self.dropout_tensor = None

    def forward(self, x: torch.Tensor, training: bool):
        if not training or self.dropout_prob == 0:
            return x
        prob_tensor = torch.full(x.shape, 1 - self.dropout_prob).to(device)
        self.dropout_tensor = torch.bernoulli(prob_tensor).to(device)
        result = self.dropout_tensor * (1 / (1 - self.dropout_prob)) * x
        return result

    def backward(self, doutput: torch.Tensor):
        if self.dropout_prob == 0:
            return doutput
        assert self.dropout_tensor is not None
        dx = doutput * self.dropout_tensor * (1 / (1 - self.dropout_prob))
        return dx

    def apply_gradient(self, learning_rate: float):
        pass


class Attention:
    def __init__(
        self, n_embed: int, n_heads: int, dropout: float, ws: Optional[AttentionWeights]
    ):
        self.x = None
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.inv_sqrt_head_size = 1.0 / math.sqrt(self.n_embed / self.n_heads)
        self.key_matrix = 2
        # TODO: check the magnitude of these values
        if ws is not None:
            self.q_map = Linear(
                ws.q_weight, ws.q_bias, True, n_embed, n_embed, False, None
            )
            self.k_map = Linear(
                ws.k_weight, ws.k_bias, True, n_embed, n_embed, False, None
            )
            self.v_map = Linear(
                ws.v_weight, ws.v_bias, True, n_embed, n_embed, False, None
            )
            self.proj_map = Linear(
                ws.proj_weight, ws.proj_bias, True, n_embed, n_embed, False, None
            )
        else:
            self.q_map = Linear(None, None, True, n_embed, n_embed, False, None)
            self.k_map = Linear(None, None, True, n_embed, n_embed, False, None)
            self.v_map = Linear(None, None, True, n_embed, n_embed, False, None)
            self.proj_map = Linear(None, None, True, n_embed, n_embed, False, None)

        self.dropout = Dropout(dropout_prob=dropout)

        self.softmax = Softmax(
            dimension=3
        )  # Sum over the keys dimension (last dimension in masked)

    def split_heads(self, qk: torch.Tensor) -> torch.Tensor:
        # input is B,T,C output is B,nh,T,hs
        return qk.view(
            qk.shape[0], qk.shape[1], self.n_heads, int(self.n_embed / self.n_heads)
        ).transpose(-3, -2)

    def combine_heads(self, qk: torch.Tensor) -> torch.Tensor:
        B = qk.shape[0]
        T = qk.shape[2]
        # input is B,nh,T,hs  output it B,T,C
        return qk.transpose(-3, -2).reshape(B, T, self.n_embed)

    def forward(self, x: torch.Tensor, training: bool):
        # X is B, T, C tensor
        # each C tensor should be multiplied by a K and V matrix, resulting in a C sized K or V vector
        # to B,T,C @ C,C matrix
        self.x = x.clone()
        self.q = self.split_heads(self.q_map.forward(x, training))  # B,nh,T,hs
        self.k = self.split_heads(self.k_map.forward(x, training)).transpose(
            -2, -1
        )  # B,nh,hs,T
        self.v = self.split_heads(self.v_map.forward(x, training))  # B,nh,T,hs
        self.dot_prods1 = (self.q @ self.k) * self.inv_sqrt_head_size  # B,nh,T,T
        # del q, k
        self.mask = torch.ones_like(self.dot_prods1).tril()
        self.dot_prods2 = self.dot_prods1.masked_fill(self.mask == 0, float("-inf"))
        # del mask

        self.attention0 = self.softmax.forward(self.dot_prods2, training)
        self.attention1 = self.dropout.forward(self.attention0, training)
        self.attention2 = self.attention1 @ self.v  # B,nh,TT x B,nh,T,hs ->  B,nh,T,hs
        self.attention3 = (
            self.attention2.transpose(1, 2)
            .contiguous()
            .view(self.attention2.shape[0], self.attention2.shape[2], self.n_embed)
        )  # B,T,C

        self.deltas = self.proj_map.forward(
            self.attention3, training
        )  # Still BTC, but now with the correct values
        # del dot_prods1, dot_prods2, attention0, attention2, attention3
        if device == "cuda":
            torch.cuda.empty_cache()

        return self.deltas

    def backward(self, ddeltas: torch.Tensor) -> torch.Tensor:
        dattention3 = self.proj_map.backward(ddeltas)  # B,T,C
        dattention2 = dattention3.view(
            dattention3.shape[0],
            dattention3.shape[1],
            self.n_heads,
            int(self.n_embed / self.n_heads),
        ).transpose(1, 2)

        q = self.split_heads(self.q_map.forward_no_cache(self.x))  # B,nh,T,hs
        k = self.split_heads(self.k_map.forward_no_cache(self.x)).transpose(
            -2, -1
        )  # B,nh,hs,T
        v = self.split_heads(self.v_map.forward_no_cache(self.x))  # B,nh,T,hs

        dattention1 = dattention2 @ v.transpose(-2, -1)
        dv = self.attention1.transpose(-2, -1) @ dattention2
        dattention0 = self.dropout.backward(dattention1)
        ddot_prods = self.softmax.backward(dattention0, self.dot_prods2.grad)

        # Grads should not flow where mask == 0
        mask = torch.ones_like(ddot_prods).tril()
        ddot_prods.masked_fill_(mask == 0, 0)

        # Do the scaling
        ddot_prods *= self.inv_sqrt_head_size

        dq = ddot_prods @ k.transpose(-2, -1)  # B,nh,T,T @ B,nh,T,hs  -> B,nh,T,hs
        dk = q.transpose(-2, -1) @ ddot_prods

        # Now we want to unsplit the heads
        dq = self.combine_heads(dq)
        dk = self.combine_heads(dk.transpose(-2, -1))
        dv = self.combine_heads(dv)

        dx1 = self.v_map.backward(dv)
        dx2 = self.k_map.backward(dk)
        dx3 = self.q_map.backward(dq)

        dx = dx1 + dx2 + dx3

        return dx

    def apply_gradient(self, learning_rate: float):
        self.proj_map.apply_gradient(learning_rate)
        self.dropout.apply_gradient(learning_rate)
        self.q_map.apply_gradient(learning_rate)
        self.k_map.apply_gradient(learning_rate)
        self.v_map.apply_gradient(learning_rate)


class TransformerBlock:
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        dropout: float,
        transformer_weights: Optional[TransformerWeights],
    ):
        if transformer_weights is None:
            self.attention_section = [
                LayerNorm(None, None, n_embed),
                Attention(n_embed, n_heads, dropout, None),
                Dropout(dropout),
            ]
            self.MLP_section = [
                LayerNorm(None, None, n_embed),
                Linear(None, None, True, n_embed, n_embed * 4, True),
                Gelu(),
                Linear(None, None, True, n_embed * 4, n_embed, True),
                Dropout(dropout),
            ]
        else:
            self.attention_section = [
                LayerNorm(
                    transformer_weights.ln_1_weight, transformer_weights.ln_1_bias, None
                ),
                Attention(n_embed, n_heads, dropout, transformer_weights.attention),
                Dropout(dropout),
            ]
            self.MLP_section = [
                LayerNorm(
                    transformer_weights.ln_2_weight, transformer_weights.ln_2_bias, None
                ),
                Linear(
                    transformer_weights.mlp.fc_weight,
                    transformer_weights.mlp.fc_bias,
                    True,
                    n_embed,
                    n_embed * 4,
                    False,
                ),
                Gelu(),
                Linear(
                    transformer_weights.mlp.proj_weight,
                    transformer_weights.mlp.proj_bias,
                    True,
                    n_embed * 4,
                    n_embed,
                    False,
                ),
                Dropout(dropout),
            ]

    def forward(self, x0: torch.Tensor, training: bool):
        self.inter0: torch.Tensor = x0.clone()
        self.inter01 = self.attention_section[0].forward(self.inter0, training)
        torch.cuda.empty_cache()
        self.inter02 = self.attention_section[1].forward(self.inter01, training)
        torch.cuda.empty_cache()
        self.inter03 = self.attention_section[2].forward(self.inter02, training)
        torch.cuda.empty_cache()
        # for layer in self.attention_section:
        #     inter0 = layer.forward(inter0, training)
        #     torch.cuda.empty_cache()

        x1 = self.inter03 + x0
        self.inter1 = x1.clone()

        self.inter2 = self.MLP_section[0].forward(self.inter1, training)
        self.inter3 = self.MLP_section[1].forward(self.inter2, training)
        self.inter4 = self.MLP_section[2].forward(self.inter3, training)
        self.inter5 = self.MLP_section[3].forward(self.inter4, training)
        self.inter6 = self.MLP_section[4].forward(self.inter5, training)
        # for layer in self.MLP_section:
        #     inter1 = layer.forward(inter1, training)
        #     torch.cuda.empty_cache()

        self.x3 = self.inter6 + x1
        return self.x3

    def backward(self, dx3: torch.Tensor) -> torch.Tensor:
        dinter6 = dx3.clone()
        dinters = [dinter6]
        dx1 = dx3.clone()

        for layer in reversed(self.MLP_section):
            dinters.append(layer.backward(dinters[-1]))
            # dinter1 = dinters[-1]
            # dinter1 = layer.backward(dinter1)

        dx1 += dinters[-1].clone()

        dinter03 = dx1.clone()
        dx0 = dx1.clone()

        dinter02 = self.attention_section[2].backward(dinter03)
        dinter01 = self.attention_section[1].backward(dinter02)
        dinter0 = self.attention_section[0].backward(dinter01)

        # for layer in reversed(self.attention_section):
        #     dinter0 = layer.backward(dinter0)
        dx0 += dinter0
        return dx0

    def apply_gradient(self, learning_rate: float):
        for layer in reversed(self.MLP_section):
            layer.apply_gradient(learning_rate)

        for layer in reversed(self.attention_section):
            layer.apply_gradient(learning_rate)
