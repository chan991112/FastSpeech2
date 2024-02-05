import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        #query와 key matrix의 행렬곱 수행
        attn = attn / self.temperature
        #그 다음 이 attention을 temperature 로 나눠서 energy를 구해준다..? 여기서 temperature란 scale이라는데..뭐지?
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        #마스크가 있다면 이렇게 마스크 값 처리 해주고, 저기서 -np.inf는 마스크가 필요없는 단어는 -inf 곱해줘서 신경쓰지 말라는 것. 근데 애초에 mask=none하고 들어감
        attn = self.softmax(attn)
        #구한 attention에 softmax 취해준다
        output = torch.bmm(attn, v)
        #output으로는 마지막으로 value 까지 곱해준다.
        return output, attn
