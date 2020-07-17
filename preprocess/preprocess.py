import os
import pandas as pd
from preprocess.functional import sentence_filter, load_label, sentence_to_target


def preprocess(dataset_path):
    print('preprocess started..')

    for directory in os.listdir(dataset_path):
        path = os.path.join(dataset_path, directory)
        for file in os.listdir(path):
            if file.endswith('.txt'):
                new_sentence = str()

                with open(os.path.join(path, file), "r") as f:
                    raw_sentence = f.read()
                    new_sentence = sentence_filter(raw_sentence)

                with open(os.path.join(path, file), "w") as f:
                    f.write(new_sentence)

            else:
                continue


def create_char_labels(dataset_path):
    print('create_char_labels started..')

    label_list = list()
    label_freq = list()

    for directory in os.listdir(dataset_path):
        path = os.path.join(dataset_path, directory)
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
            else:
                continue

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    # save to csv
    label_df = pd.DataFrame(label)
    label_df.to_csv("aihub_labels.csv", encoding="utf-8", index=False)


def create_script(dataset_path, script_prefix):
    print('create_script started..')
    char2id, id2char = load_label('aihub_labels.csv')

    for directory in os.listdir(dataset_path):
        path = os.path.join(dataset_path, directory)
        for file in os.listdir(path):
            if file.endswith('.txt'):
                sentence, target = None, None

                with open(os.path.join(path, file), "r") as f:
                    sentence = f.read()

                with open(os.path.join(path, script_prefix + file[12:]), "w") as f:
                    target = sentence_to_target(sentence, char2id)
                    f.write(target)
