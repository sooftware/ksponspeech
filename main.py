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
    parser = argparse.ArgumentParser(description='KsponSpeech Preprocess')
    parser.add_argument('--dataset_path', type=str,
                        default='E:/KsponSpeech/original',
                        help='path of original dataset')
    parser.add_argument('--new_path', type=str,
                        default='E:/KsponSpeech/character',
                        help='new path to save')
    parser.add_argument('--script_prefix', type=str,
                        default='KsponScript_',
                        help='script_prefix + FILENUM.txt : KsponScript_000001.txt')
    parser.add_argument('--labels_dest', type=str,
                        default='E:/KsponSpeech',
                        help='destination to save character labels file')
    parser.add_argument('--preprocess_method', type=str,
                        default='character',
                        help='character or subword (Will be added grapheme)')
    parser.add_argument('--character_preprocess_mode', type=str,
                        default='phonetic',
                        help='phonetic: 칠 십 퍼센트, spelling: 70%')
    parser.add_argument('--vocab_size', type=int, default=5000)
    opt = parser.parse_args()

    if opt.preprocess_method == 'character':
        preprocess(opt.dataset_path, opt.character_preprocess_mode)
        create_char_labels(opt.dataset_path, opt.labels_dest)
        create_character_script(opt.dataset_path, opt.new_path, opt.script_prefix, opt.labels_dest)
        gather_files(opt.dataset_path, opt.new_path)

    elif opt.preprocess_method == 'subword':
        generate_sentencepiece_input(opt.dataset_path)
        train_sentencepiece(opt.dataset_path, opt.vocab_size)
        create_subword_script(opt.dataset_path, opt.new_path, opt.script_prefix)
