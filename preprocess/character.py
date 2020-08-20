import os
import pandas as pd
import shutil
from preprocess.functional import (
    sentence_filter,
    load_label,
    sentence_to_target
)


def preprocess(dataset_path):
    print('preprocess started..')
    percent_files = {
        '087797': '퍼센트',
        '215401': '퍼센트',
        '284574': '퍼센트',
        '397184': '퍼센트',
        '501006': '프로',
        '502173': '프로',
        '542363': '프로',
        '581483': '퍼센트'
    }

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), "r") as f:
                        raw_sentence = f.read()
                        if file[12:18] in percent_files.keys():
                            new_sentence = sentence_filter(raw_sentence, percent_files[file[12:18]])
                        else:
                            new_sentence = sentence_filter(raw_sentence)

                    with open(os.path.join(path, file), "w") as f:
                        f.write(new_sentence)

                else:
                    continue


def create_char_labels(dataset_path, labels_dest):
    print('create_char_labels started..')

    label_list = list()
    label_freq = list()

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('txt'):
                    with open(os.path.join(path, file), "r") as f:
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

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    # save to csv
    label_df = pd.DataFrame(label)
    label_df.to_csv(os.path.join(labels_dest, "aihub_labels.csv"), encoding="utf-8", index=False)


def create_character_script(dataset_path, new_path, script_prefix, labels_dest):
    print('create_script started..')
    char2id, id2char = load_label(os.path.join(labels_dest, "aihub_labels.csv"))

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), "r") as f:
                        sentence = f.read()

                    with open(os.path.join(new_path, script_prefix + file[12:]), "w") as f:
                        target = sentence_to_target(sentence, char2id)
                        f.write(target)


def gather_files(dataset_path, new_path):
    print('gather_files started...')
    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                if file.endswith('.pcm'):
                    shutil.copy(os.path.join(path, file), os.path.join(new_path, file))
