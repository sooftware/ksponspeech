import os
import shutil
import argparse
from preprocess.preprocess import preprocess
from preprocess.character import (
    generate_character_labels,
    generate_character_script
)
from preprocess.subword import (
    generate_sentencepiece_input,
    train_sentencepiece,
    generate_subword_labels,
    generate_subword_script
)
from preprocess.grapheme import (
    character_to_grapheme,
    generate_grapheme_labels,
    generate_grapheme_script
)


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
                        help='destination to save character / subword labels file')
    parser.add_argument('--output_unit', type=str,
                        default='character',
                        help='character or subword or grapheme')
    parser.add_argument('--preprocess_mode', type=str,
                        default='phonetic',
                        help='phonetic: 칠 십 퍼센트, spelling: 70%')
    parser.add_argument('--vocab_size', type=int,
                        default=5000,
                        help='size of vocab (default: 5000)')
    parser.add_argument('--grapheme_save_path', type=str,
                        default='E:/KsponSpeech/grapheme_script',
                        help='save path of grapheme text files')
    parser.add_argument('--use_pretrain_kobert_tokenizer', '-use_pretrain_kobert_tokenizer',
                        action='store_true', default=False,
                        help='flag indication to use pretrained sentencepiece kobert tokenizer or not (default: False)')
    opt = parser.parse_args()

    preprocess(opt.dataset_path, opt.preprocess_mode)

    if opt.output_unit == 'character':
        generate_character_labels(opt.dataset_path, opt.labels_dest)
        generate_character_script(opt.dataset_path, opt.new_path, opt.script_prefix, opt.labels_dest)

    elif opt.output_unit == 'subword':
        generate_sentencepiece_input(opt.dataset_path)
        if not opt.use_pretrain_kobert_tokenizer:
            train_sentencepiece(opt.dataset_path, opt.vocab_size)
        generate_subword_labels('aihub_sentencepiece.vocab', opt.labels_dest, opt.use_pretrain_kobert_tokenizer)
        generate_subword_script(opt.dataset_path, opt.new_path, opt.script_prefix)

    elif opt.output_unit == 'grapheme':
        character_to_grapheme(opt.dataset_path, opt.grapheme_save_path)
        generate_grapheme_labels(opt.grapheme_save_path, opt.labels_dest)
        generate_grapheme_script(opt.grapheme_save_path, opt.new_path, opt.script_prefix, opt.labels_dest)

    else:
        raise ValueError("Unsupported preprocess method : {0}".format(opt.output_unit))

    gather_files(opt.dataset_path, opt.new_path)
