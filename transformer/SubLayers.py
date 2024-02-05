import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        #head 개수=scaled dot-product attention의 개수
        self.d_k = d_k
        #key dimension=d_model/n_head
        self.d_v = d_v
        #value dimension=d_model/n_head
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        #qeury의 weight, 입력 tensot의 크기가 d_model이고 출력 tensor의 크기가 n_head*d인 선형 변환을 수행하는 nn.linear 모듈 생성
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        #key의 weight
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        #value의 weight
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        #module에서 불러와서 scaled dot product attention으로 attention 만듦
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        #attention 연산 이후에 나온 head들의 attention 결과를 d_model로 묶어줌
        self.dropout = nn.Dropout(dropout)
        #drop out 비율 설정하는 것

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        #sz_b=batch size, 입력 tensor q,k,v의 입력 데이터가 한 번에 몇 개의 샘플로 구성되어 있는지
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        #q를 weight 곱해주고 이를 (batch_size, len_q, n_head, d_k)로 reshape 
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        #q.permute는 행렬을 바꾸는 것이고 뒤에 contiguous는 permute뒤에 붙어서 에러방지, view를 통해 또 reshape
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        #attention에 mask씌우기
        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """
    
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        #1차원 컨볼루션은 입력 데이터의 한 방향으로 컨볼루션 연산 수행.
        #in_channel=d_in, out_channel=d_hid, kernel_size=kernel_size[0], padding값 설정
        #입력 데이터를 더 작은 차원으로 매핑하여 추상화된 특징을 나타낼 수 있도록 설계
        
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )
        #여기서는 in_channel=d_hid, out_channel=d_in, kernel_size=kernel_size[1]
        #추상화된 특징을 다시 원래 차원으로 확장, 더 자세한 특징이나 세부사항을 추출
        self.layer_norm = nn.LayerNorm(d_in)
        #정규화해주고
        self.dropout = nn.Dropout(dropout)
        #dropout 해주고

    def forward(self, x):
        #들어온 x에 대해 forward 연산을 수행, 1D conv 거쳐서 특징 뽑아내고 residual로 빼놓은 애랑 합하는 부분
        residual = x
        output = x.transpose(1, 2)
        #1D conv 하려고 차원 교체
        output = self.w_2(F.relu(self.w_1(output)))
        #output에 대해 위에서 정의한 w_1과 w_2를 거침
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        #convolution 거친애랑 residual로 따로 빼놓은 애랑 더해서 정규화

        return output
