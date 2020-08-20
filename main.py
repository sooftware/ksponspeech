import argparse
from preprocess.character import (
    preprocess,
    create_char_labels,
    create_character_script,
    gather_files
)
from preprocess.subword import (
    generate_sentencepiece_input,
    train_sentencepiece,
    create_subword_script
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--dataset_path', type=str, default='E:/KsponSpeech/original')
    parser.add_argument('--new_path', type=str, default='E:/KsponSpeech/character')
    parser.add_argument('--script_prefix', type=str, default='KsponScript_')
    parser.add_argument('--labels_dest', type=str, default='E:/KsponSpeech')
    parser.add_argument('--preprocess_method', type=str, default='character')
    parser.add_argument('--vocab_size', type=int, default=5000)
    opt = parser.parse_args()

    if opt.preprocess_method == 'character':
        preprocess(opt.dataset_path)
        create_char_labels(opt.dataset_path, opt.labels_dest)
        create_character_script(opt.dataset_path, opt.new_path, opt.script_prefix, opt.labels_dest)
        gather_files(opt.dataset_path, opt.new_path)

    elif opt.preprocess_method == 'subword':
        generate_sentencepiece_input(opt.dataset_path)
        train_sentencepiece(opt.dataset_path, opt.vocab_size)
        create_subword_script(opt.dataset_path, opt.new_path, opt.script_prefix)
