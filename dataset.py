import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        #여기서는 AISHELL3, LibriTTS, LJSpeech 3개 사용
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        #LJSpeech에서는 "./preprocessed_data/LJSpeech" 와 같이 path 지정
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        #LJSpeech에서는 ["english_cleaners"], AISHELL3 에서는 지정없음. 특수문자 제거나 대소문자 통일 등의 작업
        self.batch_size = train_config["optimizer"]["batch_size"]
        #batch size는 16
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        #아래에서 process_meta 정의, 이걸 사용해서 각 변수에 할당
        with open(
            os.path.join(
                self.preprocessed_path, "speakers.json")
                )as f:
            self.speaker_map = json.load(f)
        #speaker.json 파일을 열어 이 json을 speaker_map에 저장     
        self.sort = sort
        self.drop_last = drop_last

    #python 내장함수로 text의 개수를 세준다.
    def __len__(self):
        return len(self.text)

    #index를 입력받아 그 안에 있는 item에 접근하는 함수로 얘도 python 내장 함수
    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        #text들을 cleaner 처리해서 phoneme으로 변환해서 담아줌
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        #실제 model에서 쓰일 mel, pitch, energy, duration 도 각각 받아줌
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        #이걸 sample 이라는 dictionaty로 묶어서 저장

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            #위에서 지정한 path로 파일 불러와 각각의 name, speaker,text, raw_text를 변수에 넣어줘서 반환
            name = []       #
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                #"|"를 기점으로 나뉨
                name.append(n)      #각 data name
                speaker.append(s)   #화자
                text.append(t)      #처리된 텍스트
                raw_text.append(r)  #원본 텍스트
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        #데이터를 1D, 2D에 맞게 padding 하는 재가공 프로세스
        #data의 idx마다 순회하면 ids, speakers, text들을 뽑아냄
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        #텍스트 길이를 하나의 array에 담음, mel도 마찬가지
        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        #data padding을 통해 input data들의 모양을 맞춰줌
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        #data를 batch 단위로 정렬하고 padding 하는 함수
        data_size = len(data)

        if self.sort:
            #sort==true 라면 len_arr은 input 데이터들의 길이를 뽑아서 내림차순 정렬
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            #sort=false 라면 그냥 data_size 만큼의 array 생성
            idx_arr = np.arange(data_size)
        #batch 크기의 배수가 되도록 마지막에 남는 애들을 tail로 따로 묶어줌
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        #처음부터 tail을 제외한 부분을 idx_arr로 다시 설정
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        #batch_size에 맞게 reshape 저기 -1은 batch_size에 맞게 알아서 설정하라는 의미
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        #drop_last 가 아니고 tail이 있으면 얘네를 따로 처리해줌
        output = list()
        #output 리스트 하나 만들고 reprocess를 통해 담아준 뒤 리턴
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

#Dataset class와 달리 train 관련된 요소들은 받지 않음, (train_config, sort, drop_last)
class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)
        #cleaner와 basename 등을 초기화 할 때 Dataset은 filename으로 불러오지만 여기서는 filepath로 불러옴
    #python 내장함수로 text의 개수를 세준다.
    def __len__(self):
        return len(self.text)
    #index를 입력받아 그 안에 있는 item에 접근하는 함수로 얘도 python 내장 함수 dataset class보다는 mel,pitch등의 인자는 없음
    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        #dataset은 sample로 묶어서 return 하지만 얘는 따로따로 return
        return (basename, speaker_id, phone, raw_text)

    #filename으로 불러와 meta data 생성, open 과정 제외하고 dataset과 같다.
    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    #data의 0번째 차원에는 id가 있고 그 만큼의 ids array 생성
    #1번째 차원에는 speaker 정보가 있고 그만큼의 array 생성
    #2번째는 text가 있고 그만큼의 array 생성
    #3번째는 raw_text가 있고 그만큼의 array 생성
    #마지막으로 text들의 len 담아주는 text_lens array 생성
    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    #improt 했을 때 모든 코드들이 실행되는 것을 방지하기 위해.
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    #parallel computing 할 수 있으면 하고 아니면 cpu에 태우고
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )
    #train set 정해주기
    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    #validation set 정해주기
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    #train_loader를 Dataloader 사용해서 객체 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    #batch가 총 몇개 들어가는지 trainset에서
    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )
    #validation set에서 batch가 총 몇개 들어가는지
    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
