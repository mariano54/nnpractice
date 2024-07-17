import dataclasses
from typing import List, Tuple, Optional
import torch
import math

device = "cuda"
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
# print(f"Using {device} device")


@dataclasses.dataclass
class LayerTorch:
    weights: torch.Tensor  # 2d Array of weights: output x input
    biases: torch.Tensor  # added to the W x a
    batch_norm: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]  # Two tensors of length output, gain and bias
    running_mean: Optional[float]
    running_var: Optional[float]


@dataclasses.dataclass
class NNWeightsTorch:
    layers: List[LayerTorch]


class ModularNetwork:
    def __init__(
        self,
        weights: Optional[NNWeightsTorch],
        momentum: float,
        B: int,
        T: int,
        C: int,
        vocab_size: int,
        n_heads: int,
        dropout: float,
        n_layers: int,
    ):
        self.max_B = B
        self.max_T = T
        self.C = C
        self.vocab_size = vocab_size
        assert weights is None  # TODO: remove
        if weights is not None:
            self.layers = [
                Linear(
                    weights.layers[0].weights,
                    weights.layers[0].biases,
                    compute_dx=False,
                ),
                LayerNorm(
                    weights.layers[0].batch_norm[0], weights.layers[0].batch_norm[1]
                ),
                Relu(),
                Dropout(0.5),
                Linear(weights.layers[1].weights, weights.layers[1].biases),
            ]
        else:
            std = 1 / math.sqrt(C)
            self.token_embeddings = torch.normal(0, std, (vocab_size, C)).to(device)
            self.positional_embeddings = torch.normal(0, std, (T, C)).to(device)
            self.initial_dropout = Dropout(dropout)
            self.transformer_blocks = [
                TransformerBlock(n_embed=C, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
            self.final_ln = LayerNorm(None, None, C)

        self.final_softmax = Softmax()

    def forward(self, xs: torch.Tensor, ys: Optional[torch.Tensor]) -> torch.Tensor:
        # xs and ys are shape (B,T)
        assert xs.shape[0] <= self.max_B
        assert xs.shape[1] <= self.max_T

        # TODO: enable inputs of different sizes within batch

        embeddings = self.token_embeddings[xs]  # Should result in B,T,C vector
        position_emb = self.positional_embeddings[torch.arange(0, xs.shape[1])]

        start = embeddings + position_emb
        B, T, C = start.shape

        filtered_start = self.initial_dropout.forward(start, True)
        inter = filtered_start
        for block in self.transformer_blocks:
            inter = block.forward(inter, True)
        inter = self.final_ln.forward(inter, True)

        inter = inter.view((B * T), C)  # B*T training examples
        logits = inter @ self.token_embeddings.T

        self.final_softmax.forward(logits, ys is not None)
        if ys is not None:
            loss = -torch.mean(
                torch.log(
                    self.final_softmax.probs[
                        range(self.final_softmax.probs.shape[0]), ys.view((B * T))
                    ]
                )
            )
            print("Loss", loss)
        return self.final_softmax.probs.view((B, T, self.vocab_size))[:, -1]

    def generate(self, xs: torch.Tensor, n_tokens: int):
        B = xs.shape[0]  # batch size
        initial_len = xs.shape[1]
        assert B <= self.max_B
        while xs.shape[1] < initial_len + n_tokens:
            last_word_probs = self.forward(xs[:, -self.max_T :], None)
            last_gen_token: List[int] = []
            for batch_index in range(B):
                one_stream_probs = last_word_probs[batch_index]
                sampled_token = torch.multinomial(one_stream_probs, num_samples=1)
                last_gen_token.append(sampled_token.item())
            xs = torch.cat((xs, torch.tensor(last_gen_token).unsqueeze(1)), dim=1)
        return xs

    def backward(self, labels: torch.Tensor) -> None:
        assert self.final_softmax.probs is not None
        loss = -torch.mean(
            torch.log(
                self.final_softmax.probs[
                    range(self.final_softmax.probs.shape[0]), labels
                ]
            )
        )
        # loss.backward()
        doutput = self.final_softmax.backward_cross_entropy(labels)
        for layer in reversed(self.layers):
            doutput = layer.backward(doutput)

    def apply_gradient(self, learning_rate: float) -> None:
        for layer in reversed(self.layers):
            layer.apply_gradient(learning_rate)

    def get_weights(self) -> NNWeightsTorch:

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
        self.last_phi = None

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        self.last_x = x
        self.last_phi = torch.tanh(math.sqrt(2 / math.pi) * (x + 0.0044715 * x**3))
        return 0.5 * x * (1 + self.last_phi)

    def backward(self, doutput: torch.Tensor) -> torch.Tensor:
        dgelu = 0.5 * (
            1
            + self.last_phi
            + self.last_x
            * math.sqrt(2 / math.pi)
            * (1 + 0.13145 * self.last_x**2)
            * (1 - self.last_phi**2)
        )
        return dgelu * doutput

    def apply_gradient(self, learning_rate: float):
        pass


class Softmax:
    def __init__(self):
        self.probs = None

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = x - torch.max(x, dim=1, keepdim=True).values
        exps = torch.exp(x)
        denominators = torch.sum(exps, dim=1, keepdim=True)
        self.probs = exps / denominators
        return self.probs

    def backward_cross_entropy(self, labels: torch.Tensor) -> torch.Tensor:
        n = self.probs.shape[0]
        d_logits = self.probs.clone()
        d_logits[range(n), labels] -= 1
        return d_logits / n

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
            std_dev = math.sqrt(2) / math.sqrt(output_dim)
            if residual_scaling_num_layers is not None:
                std_dev /= 2 * math.sqrt(residual_scaling_num_layers)

            self.weights = (
                torch.normal(mean=0, std=std_dev, size=(output_dim, input_dim))
                .float()
                .to(device)
            )
            if zero_biases:
                self.bias = torch.zeros(output_dim).float().to(device)
            else:
                self.bias = (
                    torch.normal(mean=0, std=1, size=(output_dim,)).float().to(device)
                )

        self.last_inputs = None
        self.dW = None
        self.dbias = None
        self.compute_dx = compute_dx

    def forward(self, xs: torch.Tensor, training: bool) -> torch.Tensor:
        self.last_inputs = xs
        ret = xs @ self.weights.T + self.bias
        return ret

    def backward(self, doutput: torch.Tensor) -> torch.Tensor:
        self.dW = self.last_inputs.T @ doutput
        self.dbias = doutput.sum(0)
        if self.compute_dx:
            return doutput @ self.weights
        return torch.Tensor()

    def apply_gradient(self, learning_rate: float):
        self.weights -= learning_rate * self.dW.T
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
            self.bn_gain = torch.randn((1, layer_size)).to(device) * 0.1 + 1.0
            self.bn_bias = torch.randn((1, layer_size)).to(device) * 0.1
        self.bn_var_inv = None  # Used in backprop
        self.x_hat = None  # Used in backprop
        self.d_bias = None
        self.d_gain = None

    def forward(
        self,
        z: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        bn_mean = (1.0 / z.shape[1]) * (z.sum(1, keepdim=True))  # 32x1
        self.bn_diff = z - bn_mean  # 32x1000
        self.bn_diff_sq = self.bn_diff**2  # 32x1000
        self.bn_var = (1.0 / (z.shape[1])) * self.bn_diff_sq.sum(
            1, keepdim=True
        )  # 32x1

        self.bn_var_inv = (
            self.bn_var + 1e-5
        ) ** -0.5  # This is 1/sqrt(var + epsilon)  #32x1
        self.x_hat = self.bn_diff * self.bn_var_inv  # 32x1000
        preact = self.x_hat * self.bn_gain + self.bn_bias  # 32x1000
        return preact

    def backward(self, doutput: torch.Tensor):
        d_preact: torch.Tensor = doutput
        assert self.bn_var_inv is not None
        assert self.x_hat is not None
        n = doutput.shape[0]
        layer_size = doutput.shape[1]

        d_xhatdL = self.bn_gain * d_preact
        d_bnvarinvdL = (self.bn_diff * d_xhatdL).sum(1, keepdim=True)
        d_bnvardL = (-0.5 * (self.bn_var + 1e-5) ** (-1.5)) * d_bnvarinvdL
        d_bndiffsqrdL = (1.0 / (layer_size)) * d_bnvardL.expand(n, layer_size)
        d_bndiffdL = self.bn_var_inv * d_xhatdL + (2 * self.bn_diff) * d_bndiffsqrdL
        d_bn_meandL = (-1 * d_bndiffdL).sum(1, keepdim=True)
        d_z = d_bndiffdL + (1 / layer_size) * d_bn_meandL.expand(n, layer_size)

        self.d_gain = (self.x_hat * d_preact).sum(0, keepdim=True)
        self.d_bias = d_preact.sum(0, keepdim=True)
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
            self.bn_gain = torch.randn((1, layer_size)).to(device) * 0.1 + 1.0
            self.bn_bias = torch.randn((1, layer_size)).to(device) * 0.1
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
        if not training:
            return x
        prob_tensor = torch.full(x.shape, 1 - self.dropout_prob).to(device)
        self.dropout_tensor = torch.bernoulli(prob_tensor).to(device)
        result = self.dropout_tensor * (1 / (1 - self.dropout_prob)) * x
        return result

    def backward(self, doutput: torch.Tensor):
        assert self.dropout_tensor is not None
        dx = doutput * self.dropout_tensor * (1 / (1 - self.dropout_prob))
        return dx

    def apply_gradient(self, learning_rate: float):
        pass


class Attention:
    def __init__(self, n_embed: int, n_heads: int):
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor, training: bool):
        # X is B x T x C tensor
        return x

    def backward(self, doutput: torch.Tensor):
        pass


class TransformerBlock:
    def __init__(self, n_embed: int, n_heads: int, dropout: float):
        # TODO: configure

        self.attention_section = [
            LayerNorm(None, None, n_embed),
            Attention(n_embed, n_heads),
            Dropout(dropout),
        ]

        self.MLP_section = [
            LayerNorm(None, None, n_embed),
            Linear(None, None, True, n_embed, n_embed * 4, True),
            Gelu(),
            Linear(None, None, True, n_embed * 4, n_embed, True),
            Dropout(dropout),
        ]

    def forward(self, x: torch.Tensor, training: bool):
        inter: torch.Tensor = x.clone()  # Intermediate output
        for layer in self.attention_section:
            inter = layer.forward(inter, training)

        # TODO: consider residual scaling
        inter = inter + x

        for layer in self.MLP_section:
            inter = layer.forward(inter, training)
        inter = inter + x
        return inter

    def backward(self, doutput: torch.Tensor):
        # TODO: write
        return doutput


def random_weights_nn(
    data_size: int,
    layer_sizes: List[Tuple[int, bool]],
    device: str,
    seed: Optional[int] = None,
) -> NNWeightsTorch:
    if seed is not None:
        torch.manual_seed(seed)
    activation_size = data_size

    all_layers: List[LayerTorch] = []
    for layer_num, (layer_size, has_batchnorm) in enumerate(layer_sizes):
        std_dev = math.sqrt(2) / math.sqrt(activation_size)
        weights = torch.normal(
            mean=0, std=std_dev, size=(layer_size, activation_size)
        ).float()
        if layer_num == len(layer_sizes) - 1:
            biases = torch.zeros(layer_size).float()
        else:
            biases = torch.normal(mean=0, std=1, size=(layer_size,)).float()

        batch_norm = None
        running_mean = None
        running_var = None
        if has_batchnorm:
            bn_gain = torch.randn((1, layer_size)) * 0.1 + 1.0
            bn_bias = torch.randn((1, layer_size)) * 0.1
            batch_norm = (bn_gain.to(device), bn_bias.to(device))
            running_mean = torch.zeros(layer_size).to(device)
            running_var = torch.ones(layer_size).to(device)

        all_layers.append(
            LayerTorch(
                weights.to(device),
                biases.to(device),
                batch_norm,
                running_mean,
                running_var,
            )
        )
        activation_size = layer_size
    return NNWeightsTorch(all_layers)
