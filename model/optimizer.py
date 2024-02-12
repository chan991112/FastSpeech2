import torch
import numpy as np

#loss를 줄이는 class 설계
class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        #optimizer algorithm으로 Adam 사용, parameter는 모델의 parameter들을 update,
        #betas는 이동 평균 계산에 사용되는 모멘텀 하이퍼파라미터-> local minimum에 빠지는걸 방지하도록 momentum 사용, eps는 숫자 안정성을 위한 작은 값,
        #weight_decay는 기존 손실 함수에 가중치의 제곱항을 추가하여 가중치 감소를 유도->과적합 줄임
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        #밑에서 쓰일 변수들 config에서 불러옴
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()
        #step과 learning rate update
    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()
        #모델의 gradient를 0으로 초기화, 반복할 때마다 초기화해주고 다시 grad 구함
    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        #learning rate 정하는 함수
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        #current_step에 -0.5 제곱한 것과 warmup_step에 -1.5 제곱하고 current_step 곱한 것 중 작은 것을 lr로 지정, 이 과정에서는 lr을 증가하는 방향
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        #current_step이 anneal_step보다 크면 lr에 anneal_rate 곱해줌, 만약 anneal_steps=[200,500,1000,1500] 이럴때 current_step이 700이라면 anneal_rate를 두 번 곱
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        #한번 할 때마다 current_step 1씩 올려주고 lr을 업데이트해줌
