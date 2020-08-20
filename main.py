import argparse
from preprocess.preprocess import preprocess, create_char_labels, create_script, gather_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end Speech Recognition')
    parser.add_argument('--dataset_path', type=str, default='SET YOUR KsponSpeech corpus PATH')
    parser.add_argument('--new_path', type=str, default='SET YOUR path to store preprocessed KsponSpeech corpus')
    parser.add_argument('--script_prefix', type=str, default='KsponScript_', help='default: KsponScript_FILENUM.txt')
    parser.add_argument('--labels_dest', type=str, default='SET YOUT path th store aihub_labels.csv file')
    opt = parser.parse_args()

    preprocess(opt.dataset_path)
    create_char_labels(opt.dataset_path, opt.labels_dest)
    create_script(opt.dataset_path, opt.new_path, opt.script_prefix)
    gather_files(opt.dataset_path, opt.new_path)
