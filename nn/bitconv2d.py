import torch
import torch.nn as nn
from torch import Tensor

class BitConv2d(nn.Conv2d):
    def __init__(self, *args, num_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits

        self.eps:float = 1e-5
        self.quantization_range: int = 2 ** (num_bits - 1) # Q_b in the paper


    def ste_weights(self, weights_gamma: float) -> Tensor:
        eps: float = 1e-7
        scaled_weights:Tensor = self.weight / (weights_gamma + eps)
        bin_weights_no_grad: Tensor = torch.clamp(torch.round(scaled_weights), min=-1, max=1)
        bin_weights_with_grad: Tensor = (bin_weights_no_grad - self.weight).detach() + self.weight
        return bin_weights_with_grad


    def binarize_weights(self, weights_gamma: float) -> Tensor:
        binarized_weights = self.ste_weights(weights_gamma)
        return binarized_weights


    def quantize_activations(self, _input:Tensor, input_gamma: float) -> Tensor:
        quantized_input = torch.clamp(
                _input * self.quantization_range / input_gamma,
                -self.quantization_range + self.eps,
                self.quantization_range - self.eps,
            )
        return quantized_input
    

    def dequantize_activations(self, _input: Tensor, input_gamma: float, beta: float) -> Tensor:
        return _input * input_gamma * beta / self.quantization_range


    def forward(self, _input: Tensor) -> Tensor:
        normalized_input: Tensor = nn.functional.layer_norm(_input, (_input.shape[1:]))
        input_gamma: float = normalized_input.abs().max().item()
        weight_abs_mean: float = self.weight.abs().mean().item()

        binarized_weights = self.binarize_weights(weight_abs_mean)
        input_quant = self.quantize_activations(normalized_input, input_gamma)
        output = torch.nn.functional.conv2d(
            input=input_quant,
            weight=binarized_weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        output = self.dequantize_activations(output, input_gamma, weight_abs_mean)

        return output