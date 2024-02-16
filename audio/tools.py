import torch
import numpy as np
from scipy.io.wavfile import write

from audio.audio_processing import griffin_lim

#wave form에서 mel 추출
def get_mel_from_wav(audio, _stft):
    #wave form을 -1 부터 1까지의 float 값을 갖는 tensor로 변환
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    #전처리 해준 audio를 short-time fourier Transform을 사용해 mel과 energy 추출
    melspec, energy = _stft.mel_spectrogram(audio)
    #mel 과 energy를 tensor에서 numpy 배열로 변환
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy

# 반대로 mel에서 wave form으로 변환
def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = torch.stack([mel])
    #
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    #griffin_lim 함수 사용, griffin alg는 audio_processing에 정의되어 있으면 mel에서 wave form으로 변환
    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
