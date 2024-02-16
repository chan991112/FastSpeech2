import torch
import torch.nn as nn

#loss 찾아내는 부분
class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        #pitch와 energy를 feature로 각각의 loss를 찾아냄
        self.mse_loss = nn.MSELoss()
        #논문에서 언급했듯이 Mean Squared Error 사용 pitch, duration, energy에 사용
        self.mae_loss = nn.L1Loss()
        #mean absolute error로 절댓값 오차도 따로 처리한다. mel-prediction에 사용

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        #inputs의 6번째 부터 끝까지가 각각 위에 정의한 요소들, target값들
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        #predictions로 입력되는 요소들, prediction값들과 거기에 쓰인 mask들

        src_masks = ~src_masks #마스크 반전
        mel_masks = ~mel_masks #여기도 마스크 반전
        log_duration_targets = torch.log(duration_targets.float() + 1)
        #prediction에 log_duration_predictions로 들어오니까 input duration에 log처리
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        #mel_targets는 3차원 tensor로 1차원은 그대로 두고, 2차원에서 mel의 시간 길이를 제한하도록 해당 마스크의 길이만큼으로 잘라준다. 3차원은 mel의 주파수 축
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        #사용할 mask를 전체 mask에서 길이에 맞게 잘라줌 mask는 module 생성시 "from utils.tools import get_mask_from_lengths" 이렇게 불러옴

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        #target 값들이니까 grad를 따로 구하지는 않음. target 값들은 목표값으로 변하지 않으니까. 

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)
        #pitch가 phoneme level에서의 feature라면 src_mask로 prediction과 target masking 해주고 frame level이라면 mel_mask로 masking 각 mask는 위에서 정의되어있음

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)
        #pitch와 마찬가지로 나뉨
            
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        #duration은 모두 src_mask로 masking

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        #그냥 model에서 예측한 mel
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        #transformer에서 구현한 postnet이라고 만든 mel을 보정해주는 class를 거친 mel
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        #mel은 mel_mask로 masking, unsqueeze는 unsqueeze를 써서 마지막에 1차원을 하나 더 달아줌 위에서 보면 mel은 3차원 tensor인데 mask는 2차원 tensor임
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        #mel의 loss는 그냥 mel과 postnet_mel로 따로 mae loss를 구해줌
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        #pitch, energy, duration loss 구해줌
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )
        #total loss는 다 더해준다
        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
        #구한 모든 loss들 return
