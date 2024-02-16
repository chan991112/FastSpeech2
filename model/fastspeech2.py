import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        #구조는 논문과 동일하게 encoder, variance adaptor, decoder
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        #transformer 기반이므로 transformer에서 선언한 encoder 생성
        self.encoder = Encoder(model_config)
        #modules 에서 선언한 variance_adaptor instance 생성
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        #transformer에서 선언한 decoder 생성
        self.decoder = Decoder(model_config)
        #decoder 통과해서 나온 애들을 mel_channel에 맞게 선형 변환해서 mel_spectogram 뽑아내는 linear layer
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        #postnet은 transformer 파일에서 불러옴, mel 보정해주는 역할
        self.postnet = PostNet()
        
        
        self.speaker_emb = None
        #config_file에서 multi_speaker=none 으로 설정되어 있으나 multi_speaker 일 때(다중화자) 처리
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                #파일 열어서 config에서 다중화자일 경우 화자 수 지정
                n_speaker = len(json.load(f))
            #embedding 할 화자의 수, embedding dim으로 256    
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        
        #torch에서 forward 함수는 클래스를 생성하면서 자동으로 실행되는 함수
    def forward(
        self,
        speakers,          
        texts,
        src_lens,
        max_src_len,        #batch 내 max_seq_len, 모든 텍스트 입력을 max에 맞춰서 동일하게 하기 위해
        mels=None,
        mel_lens=None,      #각 타겟 mel_len
        max_mel_len=None,   #batch 내 max_mel_len, output mel을 padding 시켜 max와 일치하게 뽑아내기 위해
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,      #pitch_control 값을 1.0으로 고정
        e_control=1.0,
        d_control=1.0,
    ):  
        #텍스트의 padding된 부분을 나타내는 mask
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        #mel의 padding된 부분을 나타내는 mask
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        #src_mask와 mel_mask 선언
        output = self.encoder(texts, src_masks)
        #encoder 통과한 output
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        #각 변수들을 variance_adaptor 통과 시킨 값으로 update
        output, mel_masks = self.decoder(output, mel_masks)
        #output, mel_masks를 decoder 통과시킨 값으로 update
        output = self.mel_linear(output)
        #output을 mel_linear 통과시킨 값으로 update
        postnet_output = self.postnet(output) + output
        #postnet_output은 postnet 통과 시킨 값으로 따로 update 기존 output은 더해줌 이게 residual connection
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
