import os
import sentencepiece as spm


def generate_sentencepiece_input(dataset_path):
    print('generate_sentencepiece_input...')

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                with open(os.path.join(path, file), "r") as f:
                    sentence = f.read()

                with open(os.path.join(dataset_path, 'aihub_vocab.txt'), 'a') as f:
                    f.write(sentence + '\n')


def train_sentencepiece(dataset_path, vocab_size):
    spm.SentencePieceTrainer.Train(
        """
        --input=%s.txt 
        --model_prefix=aihub_sentencepiece 
        --vocab_size=%s 
        --model_type=bpe 
        --max_sentence_length=9999
        --pad_id=0
        --bos_id=1
        --eos_id=2
        --unk_id=3
        """
        % (dataset_path + 'aihub_vocab.txt', str(vocab_size))
    )


def create_subword_script(dataset_path, new_path, script_prefix):
    print('create_subword_script...')

    sp = spm.SentencePieceProcessor()
    vocab_file = "aihub_sentencepiece.model"
    sp.load(vocab_file)

    for folder in os.listdir(dataset_path):
        # folder : {KsponSpeech_01, ..., KsponSpeech_05}
        path = os.path.join(dataset_path, folder)
        for subfolder in os.listdir(path):
            path = os.path.join(dataset_path, folder, subfolder)
            for file in os.listdir(path):
                with open(os.path.join(path, file), "r") as f:
                    sentence = f.read()

                encode = sp.encode_as_ids(sentence)

                with open(os.path.join(new_path, script_prefix + file[12:]), "w") as f:
                    f.write(" ".join(map(str, encode)))
