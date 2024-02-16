import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        #duration_predictor, pitch_predictor, energy_predictor는 모두 구조는 아래에서 정의한 class인 VariancePredictor와 같고 length_regulator는 따로 정의
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]
        #yaml file 보니까 default는 phoneme_level으로 정의
        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        #config file에서 "linear"로 정의
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        #config file에서 "linear"로 정의
        n_bins = model_config["variance_embedding"]["n_bins"]
        #256으로 정의
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        #얘도 다 "linear"로 정의됐으면 그냥 true 아닌가
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]
        #file 불러와서 열고 min, max값 정의
        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
            #pitch_quantization이 "log"라면 log scale로 양자화, pitch_min부터 pitch_max 까지 n_bins-1 개의 등간격 배열을 생성
            #이걸 exp 함수로 감싸서 원래의 선형 scale로 변환 후 주파수 양자화 값들을 nn.parameter로 래핑해서 모델의 학습가능한 매개변수로 취급
            #마지막 grad=flase는 매개변수들이 학습 중에 업데이트되지 않도록 고정하는 역할
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
            #quantization이 "linear"일 때로 exp 함수 빼고는 똑같다.
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )
        #energy도 pitch와 같은 구조.
            
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        #n_bins= model_config["transformer"]["encoder_hidden"]= 256 으로 놓고 n_bins 만큼의 데이터를 입력으로 받아서 각각 256차원의 벡터로 바꿔줌
        #이 embedding vector는 모델의 학습 가능한 매개변수로 역전파 과정에서 업데이트됨
    


    def get_pitch_embedding(self, x, target, mask, control):
        #여기서 pitch_embedding 값을 update
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        #target이 있으면 embedding을 target을 양자화 scale로 정한 self.pitch_bins로 나눈 bucket에 담는다. 알고리즘에 radix_sort 느낌으로 
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        #target이 없으면 control 값은 뒤에 forward 보면 1.0으로 넣어주고 그냥 prediction 값을 embedding으로
        return prediction, embedding
        #여기서 target 값이 있으면 target을 embedding 한 값과 prediction값을 return
    
    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    #pitch와 같다.

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        
        log_duration_prediction = self.duration_predictor(x, src_mask)
        #duration_prediction은 duration_predictor에서 뽑은 값

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            #length_regulator에 signal input "x"를 넣어주고 duration 달고나온 output을 다시 x로 받음
            duration_rounded = duration_target
        #duration target이 있으면 x를 duration으로 받음 
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            #log값에 지수함수 해서 원래로 만들고 이걸 반올림 한 후(d_control=1.0), min=0 으로 해서 조정
            #torch.clamp(input, min, max, out=None) -> input 값 중에 min 보다 작으면 min으로 조정, max도 마찬가지
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)


        #pitch, energy의 feature level이 phoneme일 때
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
            #입력 시그널 x에 pitch_embedding한 값을 더함. 

        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding


        #pitch, energy의 feature level이 frame일 때, phoneme일 때와 같음, mask만 달라짐, src_mask-> mel_mask
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )



class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            #input x와 duration을 묶은 expand_target을 batch에 따라 반복
            expanded = self.expand(batch, expand_target)
            #expanded는 batch에 들어간 phoneme이 몇번 반복될 지 정해서 추가함
            output.append(expanded)
            #이걸 output에 추가해주고
            mel_len.append(expanded.shape[0])
            #output의 길이 즉 음소마다의 길이를 mel_len에 더해줌
        if max_len is not None:
            output = pad(output, max_len)
            #output을 max_len에 맞게 나머지는 0으로 padding, 각 phoneme마다 output 길이가 같아지도록
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        #각 phoneme의 duration에 맞게 list 작성
        out = list()

        for i, vec in enumerate(batch):
            #enumerate를 사용해서 batch내의 data와 i(0부터 하나씩 증가) 선언
            expand_size = predicted[i].item()
            #expand_size를 predicted[i]로 즉 이 phoneme이 얼마나 지속될 것인지
            out.append(vec.expand(max(int(expand_size), 0), -1))
            #out에 vec에 expand_size만큼 차원을 늘려서. 소리는 묵음이라도 최소한 0번은 지속되므로(음수만큼 지속될 수는 없음)
        out = torch.cat(out, 0)
        #out을 tensor로 만든다.
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
    #LR을 거친 output과 mel_len을 return


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]
        #self.input_size= 256, self.filter_size 256, self.kernel= 3, self.conv_output_size= 256, self.dropout= 0.5
        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    #1D conv
                    ("relu_1", nn.ReLU()),
                    #ReLu func
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    #normalization
                    ("dropout_1", nn.Dropout(self.dropout)),
                    #dropout
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    #1D conv     2번쨰
                    ("relu_2", nn.ReLU()),
                    #ReLu func    2번째
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    #정규화       2번째
                    ("dropout_2", nn.Dropout(self.dropout)),
                    #dropout      2번째
                ]
            )
        )
        #이렇게 쌓고 마지막에 linear_layer 추가

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        #output_size를 1차원으로 선형변환

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out
        

class Conv(nn.Module):
    #단순 1D conv 계산하는 class
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
