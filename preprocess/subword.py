import os
import pandas as pd
import sentencepiece as spm
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer


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


def generate_subword_labels(vocab_path, labels_dest):
    subword_list = list()
    id_list = list()
    count = 0

    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            subword_list.append(line.split()[0])
            id_list.append(count)
            count += 1

    subword_df = pd.DataFrame({
        'subword': subword_list,
        'id': id_list
    })
    subword_df.to_csv(os.path.join(labels_dest, 'subword_labels.csv'))


def generate_subword_script(dataset_path, new_path, script_prefix, use_pretrain_kobert_tokenizer=False):
    print('create_subword_script...')

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
                    encode = sp.encode_as_ids(sentence)

                with open(os.path.join(new_path, script_prefix + file[12:]), "w", encoding='cp949') as f:
                    f.write(" ".join(map(str, encode)))
