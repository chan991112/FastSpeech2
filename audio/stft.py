import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from audio.audio_processing import (
    dynamic_range_compression,
    dynamic_range_decompression,
    window_sumsquare,
)
#audio_processing 파일에서 정의한 함수들 불러옴

class STFT(torch.nn.Module):
    #short term furie transtorm Class
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        #filter는 특정 주파수를 강조하는 것으로 length는 어떤 주파수를 강조할지
        self.hop_length = hop_length #256
        self.win_length = win_length #1024   
        self.window = window    #hann 사용
        self.forward_transform = None
        scale = self.filter_length / self.hop_length 
        #보통 1/4로 맞춤
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        #fft는 fast furie transform으로 length만큼의 단위 행렬을 만들고 FFT를 통해 계산된 Fourier Transform의 기저 함수들을 담은 행렬이 된다.
        #이러한 기저 함수들은 주파수 도메인에서 시간 도메인으로 변환될 때 사용되며, 푸리에 변환에 관련된 연산에서 활용될 수 있다.
        cutoff = int((self.filter_length / 2 + 1))
        #필터의 길이를 반으로 나눔 fft의 대칭성을 이용해서 절반만 사용, Nyquist 정리
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        #furie transform을 하면 복소수로 나타나기 때문에 실수부와 허수부를 따로 나눔
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        #정방향 basis
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )
        
        #역방향 basis, pseudo-inverse의 transfer 사용

        if window is not None:
            assert filter_length >= win_length
            #이것도 Nyquist 정리 처럼 window로 길이를 잘라놨는데 filter가 이 길이보다 더 짧아버리면 제대로 그 주파수를 읽지 못하니까
            # get window and zero center pad it to filter_length
            #window 하나 불러옴
            fft_window = get_window(window, win_length, fftbins=True)
            #주어진 filter_len에 맞추기 위해 중앙에 zero padding
            fft_window = pad_center(fft_window, filter_length)

            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        #위에 정의한 정방향,역방향 기저에 fft_window값을 넣어줌



        #이 아래부터는 퓨리에 변환을 하는 부분

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        #batch size, sample 값 초기화
        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        #data를 3차원으로 재구성, 만약 view 안에 -1이 있으면 알아서 맞추라는 뜻
        input_data = F.pad(
            input_data.unsqueeze(1),
            #앞,뒤로 filter_len(=window_size) /2 만큼, 위 아래로는 없이 reflect 하게 smooth한 padding 추가
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        #1d conv 수행, input_data에 대해 forward_basis를 필터로 수행
        forward_transform = F.conv1d(
            input_data.cuda(),
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),
            #conv의 보폭을 hop_length로 설정
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        #실수, 허수 따로 수행
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        #magnitude에 실수 제곱, 허수 제곱을 더한 뒤 루트 씌워서 벡터의 크기를 저장
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        #복소수 평면상에 실수, 허수 간의 각도를 계산해서 phase에 저장
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    #magnitude와 phase를 입력으로 원래의 시간 영역의 audio 복원
    def inverse(self, magnitude, phase):
        #magnitude 와 phase를 사용하여 다시 복소수 형태로 concat
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        #transpose1d 를 사용하여 conv 수행 여기서 위에 정의한 inverse_basis를 필터로 사용
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        #audio에서 mel로 변환할 때의 window의 영향을 보정
        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            #window의 영향을 적게 받은 부분은 제외하고 역필터를 적용(아래 inverse_transform 연산내에 범위에 쓰임)
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
       
        #역변환 결과를 원본 input의 크기와 같도록 변환
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform


    def forward(self, input_data):
        #magnitude 와 phase를 transform 한 값으로 받음
        self.magnitude, self.phase = self.transform(input_data)
        #reconstruction도 위에 정의한 inverse 취한 값으로 받아서 return
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        #위에 만들어 놓은 STFT class 객체 하나 만듦
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        #mel filter 생성후 tensor로 변환
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
    #dynamic_range_compression을 통해 data 중에 너무 큰값은 좀 줄여주는 식, study 할 때 했던 compression과 동일
    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output
    #compression의 반대
    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        #위에서 정의한 transform 사용
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        #filter인 mel_basis와 행렬곱 수행
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        #compressor 사용
        mel_output = self.spectral_normalize(mel_output)
        #energy는 magnitude를 normalization 한 값으로 받아 return 
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy
