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
from preprocess.preprocess import preprocess, create_char_labels, create_script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--dataset_path', type=str, default='SET YOUR KsponSpeech corpus PATH')
    parser.add_argument('--script_prefix', type=str, default='KsponScript_', help='default: KsponScript_FILENUM.txt')
    opt = parser.parse_args()

    preprocess(opt.dataset_path)
    create_char_labels(opt.dataset_path)
    create_script(opt.dataset_path, opt.script_prefix)
