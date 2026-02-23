"""ScalarModel (SQ Codec) ported to MLX.

PyTorch legacy uses (N, C, L) layout for Conv1d. MLX nn.Conv1d expects (N, L, C).
We keep the same logical structure but adapt the dimension ordering.

Weight-norm is baked into the weights at conversion time, so the MLX modules use
plain Conv1d / ConvTranspose1d with the *effective* weight already stored.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


# ---------------------------------------------------------------------------
# Activation: Snake
# ---------------------------------------------------------------------------


class Snake1d(nn.Module):
    """Snake activation: x + (1/alpha) * sin(alpha * x)^2.

    Parameter ``alpha`` has shape (1, 1, C) for MLX (N, L, C) layout.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))
        self._alpha_recip: mx.array | None = None

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha
        # Lazy-cache reciprocal (alpha is a loaded weight, stable after load_weights)
        if self._alpha_recip is None:
            self._alpha_recip = (alpha + 1e-9).reciprocal()
        return x + self._alpha_recip * mx.power(mx.sin(alpha * x), 2)


# ---------------------------------------------------------------------------
# CausalConv1d  /  CausalConvTranspose1d
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding.

    In PyTorch the legacy code pads the time dim on the left by
    ``dilation * (kernel_size - 1)``.  MLX Conv1d expects (N, L, C).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, C) -- pad on the time axis (axis=1)
        x = mx.pad(x, pad_width=[(0, 0), (self.left_padding, 0), (0, 0)])
        return self.conv(x)


class Conv1d(nn.Module):
    """Wrapper matching legacy Conv1d: supports causal *or* symmetric padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        if causal:
            self.impl = CausalConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            if padding is None:
                padding = _get_padding(kernel_size, dilation)
            self.impl = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )

    def __call__(self, x: mx.array) -> mx.array:
        return self.impl(x)


class CausalConvTranspose1d(nn.Module):
    """ConvTranspose1d with causal trimming.

    Legacy assertion: kernel_size == 2 * stride, padding == 0.
    After the transposed conv the last ``stride`` samples are trimmed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride_val = stride
        self.conv_t = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_t(x)
        # Trim the last stride samples (causal)
        if self.stride_val > 0:
            x = x[:, : -self.stride_val, :]
        return x


class ConvTranspose1d(nn.Module):
    """Wrapper matching legacy ConvTranspose1d: causal or symmetric."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        if causal:
            self.impl = CausalConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                bias=bias,
            )
        else:
            padding = (kernel_size - stride) // 2
            self.impl = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

    def __call__(self, x: mx.array) -> mx.array:
        return self.impl(x)


# ---------------------------------------------------------------------------
# PReLU (MLX doesn't ship one)
# ---------------------------------------------------------------------------


class PReLU(nn.Module):
    """Parametric ReLU. Weight shape (C,) broadcast over (N, L, C)."""

    def __init__(self, num_parameters: int = 1):
        super().__init__()
        self.weight = mx.ones((num_parameters,)) * 0.25

    def __call__(self, x: mx.array) -> mx.array:
        return mx.where(x >= 0, x, self.weight * x)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PreProcessor(nn.Module):
    def __init__(
        self, n_in: int, n_out: int, num_samples: int, kernel_size: int = 7, causal: bool = False
    ):
        super().__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = PReLU()

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, C)
        x = self.activation(self.conv(x))
        # AvgPool1d with kernel=num_samples, stride=num_samples
        N, L, C = x.shape
        # Trim to exact multiple
        new_L = (L // self.num_samples) * self.num_samples
        x = x[:, :new_L, :]
        x = x.reshape(N, new_L // self.num_samples, self.num_samples, C)
        x = x.mean(axis=2)
        return x


class PostProcessor(nn.Module):
    def __init__(
        self, n_in: int, n_out: int, num_samples: int, kernel_size: int = 7, causal: bool = False
    ):
        super().__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = PReLU()

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, C) -- repeat each time step num_samples times
        # Legacy: x = x.repeat(1, 1, num_samples).view(B, -1, C)  (after transpose)
        # In (N,L,C) layout: repeat along L dimension
        N, T, C = x.shape
        x = mx.repeat(x, repeats=self.num_samples, axis=1)
        x = self.activation(self.conv(x))
        return x


class ResidualUnit(nn.Module):
    def __init__(
        self, n_in: int, n_out: int, dilation: int, res_kernel_size: int = 7, causal: bool = False
    ):
        super().__init__()
        self.conv1 = Conv1d(
            n_in, n_out, kernel_size=res_kernel_size, dilation=dilation, causal=causal
        )
        self.conv2 = Conv1d(n_in, n_out, kernel_size=1, causal=causal)
        self.activation1 = PReLU()
        self.activation2 = PReLU()

    def __call__(self, x: mx.array) -> mx.array:
        out = self.activation1(self.conv1(x))
        out = self.activation2(self.conv2(out))
        return out + x


class DownsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        self.activation = PReLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.activation(self.conv(x))


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation: bool = True,
    ):
        super().__init__()
        self.conv_t = ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            causal=causal,
        )
        self.has_activation = activation
        # Note: legacy UpsampleLayer in decoder is created with activation=None
        # so no PReLU for decoder upsample blocks.

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv_t(x)


class ResEncoderBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        stride: int,
        down_kernel_size: int,
        res_kernel_size: int = 7,
        causal: bool = False,
    ):
        super().__init__()
        self.convs = [
            ResidualUnit(
                n_in, n_out // 2, dilation=1, res_kernel_size=res_kernel_size, causal=causal
            ),
            ResidualUnit(
                n_out // 2, n_out // 2, dilation=3, res_kernel_size=res_kernel_size, causal=causal
            ),
            ResidualUnit(
                n_out // 2, n_out // 2, dilation=5, res_kernel_size=res_kernel_size, causal=causal
            ),
            ResidualUnit(
                n_out // 2, n_out // 2, dilation=7, res_kernel_size=res_kernel_size, causal=causal
            ),
            ResidualUnit(
                n_out // 2, n_out // 2, dilation=9, res_kernel_size=res_kernel_size, causal=causal
            ),
        ]
        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal
        )

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.convs:
            x = conv(x)
        return self.down_conv(x)


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        stride: int,
        up_kernel_size: int,
        res_kernel_size: int = 7,
        causal: bool = False,
    ):
        super().__init__()
        self.up_conv = UpsampleLayer(
            n_in,
            n_out,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
            activation=False,
        )
        self.convs = [
            ResidualUnit(n_out, n_out, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x


# ---------------------------------------------------------------------------
# round_func9 (inference only -- no STE needed)
# ---------------------------------------------------------------------------


def round_func9(x: mx.array) -> mx.array:
    """Scalar quantisation: round(9*x)/9."""
    return mx.round(9.0 * x) / 9.0


# ---------------------------------------------------------------------------
# ScalarModel
# ---------------------------------------------------------------------------


class ScalarModel(nn.Module):
    """The SQ-Codec encoder + decoder with scalar VQ.

    Convolutions use MLX (N, L, C) layout.
    """

    def __init__(
        self,
        num_bands: int,
        sample_rate: int,
        causal: bool,
        num_samples: int,
        downsample_factors: list[int],
        downsample_kernel_sizes: list[int],
        upsample_factors: list[int],
        upsample_kernel_sizes: list[int],
        latent_hidden_dim: int,
        default_kernel_size: int,
        delay_kernel_size: int,
        init_channel: int,
        res_kernel_size: int,
    ):
        super().__init__()

        # ---------- Encoder ----------
        encoder: list[nn.Module] = []
        encoder.append(
            Conv1d(num_bands, init_channel, kernel_size=default_kernel_size, causal=causal)
        )
        if num_samples > 1:
            encoder.append(
                PreProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        for i, down_factor in enumerate(downsample_factors):
            encoder.append(
                ResEncoderBlock(
                    int(init_channel * (2**i)),
                    int(init_channel * (2 ** (i + 1))),
                    down_factor,
                    downsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        encoder.append(
            Conv1d(
                int(init_channel * (2 ** len(downsample_factors))),
                latent_hidden_dim,
                kernel_size=default_kernel_size,
                causal=causal,
            )
        )
        self.encoder = encoder

        # ---------- Decoder ----------
        decoder: list[nn.Module] = []
        # "look ahead" conv (non-causal in legacy)
        decoder.append(
            Conv1d(
                latent_hidden_dim,
                int(init_channel * (2 ** len(upsample_factors))),
                kernel_size=delay_kernel_size,
                causal=False,  # legacy uses default Conv1d (non-causal)
            )
        )
        for i, upsample_factor in enumerate(upsample_factors):
            decoder.append(
                ResDecoderBlock(
                    int(init_channel * (2 ** (len(upsample_factors) - i))),
                    int(init_channel * (2 ** (len(upsample_factors) - i - 1))),
                    upsample_factor,
                    upsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        if num_samples > 1:
            decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        decoder.append(
            Conv1d(init_channel, num_bands, kernel_size=default_kernel_size, causal=causal)
        )
        self.decoder = decoder

    def encode(self, x: mx.array) -> mx.array:
        """Encode waveform to latent. x: (N, L, C)."""
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = mx.tanh(layer(x))
        return x

    def decode(self, x: mx.array) -> mx.array:
        """Decode latent to waveform. x: (N, L, C).

        Applies round_func9 (scalar VQ) then runs the decoder stack.
        """
        x = round_func9(x)
        for layer in self.decoder:
            x = layer(x)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        x = self.encode(x)
        x = round_func9(x)
        for layer in self.decoder:
            x = layer(x)
        return x
