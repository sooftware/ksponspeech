import os
import pandas as pd
from tqdm import tqdm
from collections import Counter


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding='utf-8')

    id_list = ch_labels['id']
    char_list = ch_labels['char']
    freq_list = ch_labels['freq']

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        target += (str(char2id[ch]) + ' ')

    return target[:-1]


def generate_character_labels(transcripts, labels_dest):
    print('create_char_labels started..')

    # build character counter
    counter = Counter([chr for transcript in transcripts for chr in transcript])
    sorted_counter = counter.most_common(n=None)

    # sort together Using zip
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (ch, freq) in enumerate(sorted_counter):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    # save to csv
    os.makedirs(labels_dest, exist_ok=True)
    label_df = pd.DataFrame(label)
    label_df.to_csv(os.path.join(labels_dest, 'aihub_labels.csv'), encoding='utf-8', index=False)


def generate_character_script(audio_paths, transcripts, labels_dest, save_path):
    print('create_script started..')
    char2id, id2char = load_label(os.path.join(labels_dest, 'aihub_labels.csv'))

    with open(os.path.join(save_path, 'transcripts.txt'), 'w') as f:
        for audio_path, transcript in tqdm(list(zip(audio_paths, transcripts)), desc='Character preprocessing...'):
            char_id_transcript = sentence_to_target(transcript, char2id)
            audio_path = audio_path.replace('txt', 'pcm')
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n')
