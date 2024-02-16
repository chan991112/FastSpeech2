import torch
import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window


def window_sumsquare(
    window,
    #window size는 사람이 소리를 짧은 시간단위로 쪼개서 듣는것에 착안하여 시계열 데이터를 일정한 시간 구간(window size)로 나누고 각 구간에 대해서 스펙트럼을 구함
    #input signal이 너무 길어서 한번에 처리하지 않고 쪼개주는 함수, Relu function과 비슷한 함수로 일부구간에서는 특정 값을 갖고 나머지에서는 0을 갖는 형태로 얘를 원래 신호에 곱하면 특정구간 외에는 모두 0이 된다.
    #w(n) : Window function 일반적으로 Hann window 사용한다. 그냥 0,1로만 박으면(rectangular window) 정확하지 않고 main lobe와 side lobe가 생기면서 오차가 심해짐, 잘라서 쓰는거다 보니 window를 사용하면 side lobe가 생기기는 함     
    n_frames,
    #몇 프레임으로 나누는가 n_frames = 1 + int((len(y) - frame_length) / hop_length)
    hop_length,
    #window가 겹치는 사이즈, 일반적으로는 1/4 정도를 겹치게 한다. frame의 양 끝단에서 신호의 정보가 자연스럽게 연결되게 하기 위해 겹치게 잡음, 각 frame의 시작점에서 hop_length만큼 띄어서 다음 frame의 시작점으로 잡음
    win_length,
    #window의 길이로 window 함수에 들어가는 sample의 양, fft size와 같으면 좋다고 한다
    n_fft,
    #window를 얼마나 많은 주파수 밴드로 나누는가 furie 분석할 때
    #n_fft : length of the windowed signal after padding with zeros. 즉, 각 frame은 window_length만큼 windowing된 후 n_fft만큼 zero-padding된다.

    dtype=np.float32,
    norm=None,
    #정규화 지정 매개변수
):
    

    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft
        #win length가 설정이 안 되어있다면 위에서 언급했듯 fft와 같게 설정

    n = n_fft + hop_length * (n_frames - 1)
    #n_frames = 1 + int((len(y) - frame_length) / hop_length)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    #window함수를 불러온다. window 종류와 win_length를 변수로 넣어줌
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    #정규화해주고 square 계산
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    #n_fft만큼 zero padding


    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x
    #위에서 output 값으로 만들어놓은 x에 값을 넣어주는 함수, frame 수만큼 반복하는 함수로 sample은 각 frame의 시작점.
    # 2 dimenstion x에 x[sample:min(~)] 부분에 win_sq한 값을 넣어줌, x는 0으로 초기화되어 있음

def griffin_lim(magnitudes, stft_fn, n_iters=30):
    #melspectogram 으로 계산된 magnitude에서 실제 audio를 만들어내는 알고리즘
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)
#입력신호 x에 대해 압축해주는 함수, 소리의 세기를 균일하게 조절하거나 특정 빈도 대역의 강도를 강조하는 데 활용
#torch.clamp: input tensor x의 값을 clip_val보다 작은 값으로 제한,
#압축 계수 C를 통해 log값 조절
#제한된 값에 대해 로그를 취해 큰값을 작게 만들고 동적 범위 압축

def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C
#위 함수의 반대로, 지수취해주고 C로 나눠줌
