import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#parallel programming 위해서 cuda가 가능하면 cuda, serial은 cpu

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    #batch_size=16
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    #dataset의 길이가 64보다 길면 batch_size=64
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    #데이터 로드 완료


    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    #fastspeech model과 optimizer 불러오기
    model = nn.DataParallel(model)
    #모델을 병렬로 실행해 다수의 gpu에서 작동하도록 하는 code
    num_param = get_param_num(model)
    #파라미터의 총 개수
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    #loss file에 구현해놓은 loss 정의
    print("Number of FastSpeech2 Parameters:", num_param)


    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    #utils에 구현되어 있는 get_vocoder

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    #각 ckpt_path, log_path, result_path 들에 대해 학습 log를 만들어주려고 dir 생성
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    #log_path 밑에 sub로 "train","val"을 생성
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    #각각 dir 생성
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    #log 기록자 생성


    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    #1
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    #1.0
    total_step = train_config["step"]["total_step"]
    #900000
    log_step = train_config["step"]["log_step"]
    #100
    save_step = train_config["step"]["save_step"]
    #100000
    synth_step = train_config["step"]["synth_step"]
    #1000
    val_step = train_config["step"]["val_step"]
    #1000

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    #tqdm을 사용해서 전체 step에 대한 진행도 나타내는 부분

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        #epoch당 진행도 나타내는 tqdm
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                #batch를 gpu에 전달

                # Forward
                output = model(*(batch[2:]))
                #batch를 model 돌림
                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                #total_loss를 grad_acc_step으로 나눔. 
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    #grad_acc_step=1 이므로 매 step 마다 parameter update
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    #grad가 폭주하지 않도록 thresh(임계값)을 지정
                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                    #수정 후 grad는 0으로 초기화

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )
                    #log 스텝(100)마다 현재 step과 각 loss들을 message에 담음 
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                        #이 message들을 log_path dir에 담음
                    outer_bar.write(message1 + message2)
                    #프롬프트 창에 띄움
                    log(train_logger, step, losses=losses)
                    

                if step % synth_step == 0:
                    #synth_step(1000)마다 합성
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    #utils-tools 파일 보면 fig는 pred, ground_truth 의 두가지 mel을 담는다. wav_reconstruction은 target을 vocoder에 넣어서 만든 wave form, wav_prediction은 prediction을 wavform으로 만든 것
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    #config logging
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    #wav_reconstruction logging
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )
                    #wav_prediction logging

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()
                    #val_step 마다 evaluate 불러와서 validation log에 저장, log_step, synth_step에 나타나는 log함수를 사용하는 건 evaluate 함수내에서 나타남
                    
                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )
                    #save step마다 save

                if step == total_step:
                    quit()
                step += 1
                #step 관리
                outer_bar.update(1)
                #진행률 update

            inner_bar.update(1)
            #진행률 update
        epoch += 1
        #epoch 관리



if __name__ == "__main__":
    #evaluate.py과 같다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
