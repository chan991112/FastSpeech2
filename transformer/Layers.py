from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(torch.nn.Module):
    """FFT Block"""
    #feed-forward Transformer
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #self attention 선언
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )
        #positionwise_ffn 하나 선언해줌

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        #enc_input에 대해서 self_attention 구함, 그리고 이걸 enc_output, enc_slf_attn에 저장
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        #mask 처리 해줌, 계산의 효율 증가
        enc_output = self.pos_ffn(enc_output)
        #output을 positionwise ffn 한번 돌림
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn
        #return은 self attention과 positional ffn 통과한 enc_output, self_attn

class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        #conv kernel size
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        #padding이 없을 때 padding 넣어주기
            
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        #1D conv 선언

    def forward(self, signal):
        conv_signal = self.conv(signal)
        
        return conv_signal
        #1D conv 처리 해준 signal return

class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """
    #mel spectogram에서 생성된 음성 파형의 세부 정보를 보완하는 역할을 한다

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
    #초기화 변수: mel-spec channel, embedding dim, kernel size, conv 수
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()
        #convolution 5개라고 위에서 초기화 했는데 얘네를 리스트로 묶어서 관리

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )
        #convolutions에 위에서 정의한 ConvNorm 넣어주고, 정규화 모듈인 nn.BatchNorm1d도 하나 추가
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )
        #위에랑 같이 안 해주고 따로 빼서 이렇게 나머지 3개의 모듈을 따로 넣어주는 건 위에서는 out_channel= postnet_embedding_dim로 나와서
        #dim 맞춰주려고 따로 넣어줌    
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )
        #마지막 하나의 모듈은 input_channel은 postnet_embedding_dim으로 ouput_channel은 원래 n_mel_channels로 뱉어줌

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        #input x에 대해서 위에서 정의한 convolutions를 통과하도록 만들어줌
        x = x.contiguous().transpose(1, 2)
        return x
