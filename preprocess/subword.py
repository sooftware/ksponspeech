import os
import pandas as pd
import sentencepiece as spm
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer


def load_label(filepath):
    grpm2id = dict()
    id2grpm = dict()

    grapheme_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = grapheme_labels["id"]
    grapheme_list = grapheme_labels["grapheme"]

    for (idx, grapheme) in zip(id_list, grapheme_list):
        grpm2id[grapheme] = idx
        id2grpm[idx] = grapheme

    return grpm2id, id2grpm


def sentence_to_target(sentence, subword2id):
    target = str()

    for grapheme in sentence:
        target += (str(subword2id[grapheme]) + ' ')

    return target[:-1]


def generate_sentencepiece_input(dataset_path):
    print('generate_sentencepiece_input...')

    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                with open(os.path.join(path, file), "r") as f:
                    sentence = f.read()

                with open(os.path.join(dataset_path, 'aihub_vocab.txt'), 'a', encoding='cp949') as f:
                    f.write(sentence + '\n')


def train_sentencepiece(dataset_path, vocab_size):
    spm.SentencePieceTrainer.Train(
        '--input=%s.txt '
        '--model_prefix=aihub_sentencepiece '
        '--vocab_size=%s '
        '--model_type=bpe '
        '--max_sentence_length=9999 '
        '--hard_vocab_limit=false'
        % (dataset_path + '/aihub_vocab', str(vocab_size))
    )


def generate_subword_labels(dataset_path, labels_dest):
    print('generate_subword_labels started..')

    label_list = list()
    label_freq = list()

    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('txt'):
                    with open(os.path.join(path, file), "r", encoding='cp949') as f:
                        sentence = f.read()

                        for ch in sentence:
                            if ch not in label_list:
                                label_list.append(ch)
                                label_freq.append(1)
                            else:
                                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (subword, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['subword'].append(subword)
        label['freq'].append(freq)

    # save to csv
    label_df = pd.DataFrame(label)
    label_df.to_csv(os.path.join(labels_dest, "aihub_subword_labels.csv"), encoding="utf-8", index=False)


def sentence_to_subwords(dataset_path, subword_save_path, script_prefix, use_pretrain_kobert_tokenizer=False):
    print('sentence_to_subwords...')

    if use_pretrain_kobert_tokenizer:
        tok_path = get_tokenizer()
        sp = SentencepieceTokenizer(tok_path)

    else:
        sp = spm.SentencePieceProcessor()
        vocab_file = "aihub_sentencepiece.model"
        sp.load(vocab_file)

    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                with open(os.path.join(path, file), "r", encoding='cp949') as f:
                    sentence = f.read()

                if use_pretrain_kobert_tokenizer:
                    encode = sp(sentence)
                else:
                    encode = sp.encode(sentence)

                with open(os.path.join(subword_save_path, script_prefix + file[12:]), "w", encoding='cp949') as f:
                    f.write(" ".join(map(str, encode)))


def generate_subword_script(dataset_path, new_path, script_prefix, labels_dest):
    print('generate_subword_script started..')
    subword2id, id2subword = load_label(os.path.join(labels_dest, 'aihub_subword_labels.csv'))

    for folder in os.listdir(dataset_path):
        if not folder.startswith('KsponSpeech'):
            continue
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), "r", encoding='cp949') as f:
                        sentence = f.read()

                    with open(os.path.join(new_path, script_prefix + file[12:]), "w", encoding='cp949') as f:
                        target = sentence_to_target(sentence, subword2id)
                        f.write(target)
