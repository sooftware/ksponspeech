"""
@github{
  title = {KsponSpeech.preprocess},
  author = {Soohwan Kim},
  publisher = {GitHub},
  url = {https://github.com/sooftware/KsponSpeech.preprocess},
  year = {2020}
}
"""
import argparse
from preprocess.preprocess import preprocess, create_char_labels, create_script, gather_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--dataset_path', type=str, default='SET YOUR KsponSpeech corpus PATH')
    parser.add_argument('--new_path', type=str, default='SET YOUR path to store preprocessed KsponSpeech corpus')
    parser.add_argument('--script_prefix', type=str, default='KsponScript_', help='default: KsponScript_FILENUM.txt')
    parser.add_argument('--mode', type=str, default='CHOOSE : phonetic or numeric',
                        help='default: phonetic(6->"육"), optional: numeric(6->"6")')
    parser.add_argument('--filenum_adjust', type=bool, default=False, help='adjust file number for handling "%"')
    opt = parser.parse_args()

    preprocess(opt.dataset_path, opt.mode, opt.filenum_adjust)
    create_char_labels(opt.dataset_path)
    create_script(opt.dataset_path, opt.script_prefix)
    gather_files(opt.dataset_path, opt.new_path, opt.script_prefix)
