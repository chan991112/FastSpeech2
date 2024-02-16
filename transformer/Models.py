import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

#positional encoding 부분 자세한 설명은 생략...
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

#encoder 부분
class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        #max seq의 단어개수+1
        n_src_vocab = len(symbols) + 1
        #source에 있는 어휘 총 개수
        d_word_vec = config["transformer"]["encoder_hidden"]
        #encoder의 word embedding 차원: 256
        n_layers = config["transformer"]["encoder_layer"]
        #encoder layer의 수: 4
        n_head = config["transformer"]["encoder_head"]
        #head의 개수: 서로 다른 attention 컨셉의 수: 2
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        #attention의 key와 value 차원: 256/2
        d_model = config["transformer"]["encoder_hidden"]
        #임베딩 차원: 256
        d_inner = config["transformer"]["conv_filter_size"]
        #feedforward network inner layer의 차원: 1024
        kernel_size = config["transformer"]["conv_kernel_size"]
        #feedforward network convolution kernel의 크기
        dropout = config["transformer"]["encoder_dropout"]
        #drop out 비율: [9:1]
        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        #단어 임베딩하기
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        #positional encoding 불러와서 position 정보 넣어주기
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        # n_layer만큼 FFT block 쌓아주기. n_layer 개 만큼 거쳐서 마지막 layer에서 나온 결과를 decoder로 보내줌

    def forward(self, src_seq, mask, return_attns=False):
        #입력 시퀀스를 인코딩하여 출력을 생성하는 함수
        enc_slf_attn_list = []
        #self attention_list 만들어두기
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        #여기서는 source seq 기준으로 batch_size, max_len 결정해줌
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        #self_attention mask 준비

        # -- Forward



        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
            #모델이 학습 중이 아니고, 입력 seq의 길이가 최대보다 크면
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
        # 워드 임베딩과 sin을 이용한 위치 인코딩해주기
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            #layer stack에 있는 layer들에 대해 계속 다음 layer로 결과를 보내주는 과정
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
            #계속해서 attention 결과 저장
        return enc_output
            #마지막 layer의 output만 return
    
#decoder 부분
class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]#256
        n_layers = config["transformer"]["decoder_layer"]   #6
        n_head = config["transformer"]["decoder_head"]      #2
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]         #256//2 key와 value의 dim=hidden//head
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]  #256
        d_inner = config["transformer"]["conv_filter_size"]#1024
        kernel_size = config["transformer"]["conv_kernel_size"] #[9,1] convolution 연산 할 때 9개에 대해서 1개의 값으로 뱉음
        dropout = config["transformer"]["decoder_dropout"] #0.2 dropout은 overfitting을 막기 위해 20%의 뉴런을 다음 layer로 넘기지 않는다

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        #decoder에도 positional coding해서 넣어줌
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
                #FFTBlock layer 수 만큼 쌓아줌
            ]
        )
        #여기까지 구조는 encoder와 똑같음

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        #여기서는 enc_seq기준으로 batch_size, max_len 결정해줌


        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            #학습 중이 아니고 입력 시퀀스의 길이가 최대 시퀀스의 길이보다 클 경우
            #encoder랑 다르게 이런 마스크를 미리 하나 만듦
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
            #인코딩된 시퀀스에 위치 인코딩 더해줌
        else:
            max_len = min(max_len, self.max_seq_len)
            #max_len으로 현재 길이와 최대 길이 중 작은 값으로 선택

            # -- Prepare masks
            #attention 할 때 이후의 값을 0으로 설정해주는 마스크
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            #else 문에서도 mask 선언
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            # enc_seq[:, :max_len, :]의 의미: 우선 얘는 3D tensor임.
            #첫 번째 축(batch)은 모두 선택, 두번째 축 (시퀀스 길이)에서는 처음부터 max_len 까지 선택하고, 세번째 축에서는 모두 선택한다.
            #길이 맞춰주려고 넣은 padding 값을 무시하려고 사용하는 mask
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]
        #여기까지 decoder에 encoder에서 온 정보 positional encoding 해주고 넣어주는 과정
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            #이렇게 decoder layer 순회함.
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
            #각 layer마다 attention 결과를 갱신
        return dec_output, mask
       
