import re
import argparse
from string import punctuation
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader


from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#parallel위한 코드

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon
#lex_path에서 불러와서 lexicon이라는 list에 word를 key로 phoneme을 value로 하는 dictionary 생성, if 문은 lexicon에 이미 있는 word는 pass해서 중복되지 않도록


def preprocess_english(text, preprocess_config):
    #영어 텍스트를 전처리하여 음성 합성을 위한 text seq로 변환
    text = text.rstrip(punctuation)
    #text에서 punctuation 제거
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    #lexicon 생성
    g2p = G2p()
    #grapheme to phoneme으로 철자에서 단어의 발음을 예측하는 model
    phones = []
    #해당 text의 phoneme을 담을 phones list 생성
    words = re.split(r"([,;.\-\?\!\s+])", text)
    #word단위로 text를 분할
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
            #word가 lexicon에 있으면 phones에 추가
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
            #없으면 g2p를 이용해서 phoneme 만들어서 추가
    phones = "{" + "}{".join(phones) + "}"
    #생성된 seq들을 "{}"로 묶고 공백을 추가하여 문자열로 결합
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    #"{" 로 시작하는 문자열을 찾고 숫자가 공백이 아닌 문자 또는 공백이 0개 또는 1개인 부분을 찾고 끝이 "}"인 부분을 찾는다.
    #-> "{}"로 묶인 부분을 찾아 "{sp}"로 대체,
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    #이를 음성합성을 위해 시퀀스로 변환, 그냥 phones랑 text 반환하면 되는거 아닌가..?

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    #중국어 텍스트를 pinyin(병음)으로 반환한 결과를 담은 list. 즉 한자 발음을 로마자로 표기한 것. 그리고 이게 lexicon에 있는지 확인하고 phones에 추가, 이후 과정은 영어와 같음
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    #input으로 들어간 config를 전처리, 모델, 학습 config에 넣어줌
    pitch_control, energy_control, duration_control = control_values
    #합성할 때 사용할 pitch, energy, duration 요소들을 control_values라는 input으로 받아서 저장
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            #여기서는 변할 값이 없으니까 메모리 아끼기 위해 no_grad

            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            #합성해주는 모델에 batch와 control 요소들을 입력해서 prediction을 output으로 받음
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
            #prediction을 직접 vocoder를 이용해 합성해서 wav form을 만듦




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
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
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()
# 여기까지는 train과 마찬가지로 argumentparser를 이용해 arg 추가
    

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None


    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    #config들 읽어오기.

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)


    #batch랑 single이 뭐지..?
    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
