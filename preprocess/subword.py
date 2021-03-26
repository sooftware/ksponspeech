import os
import sentencepiece as spm
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


def train_sentencepiece(transcripts, datapath: str = './data', vocab_size: int = 5000):
    print('generate_sentencepiece_input..')

    if not os.path.exists(datapath):
        os.mkdir(datapath)

    with open(f'{datapath}/sentencepiece_input.txt', 'w') as f:
        for transcript in transcripts:
            transcript = transcript.upper()
            f.write(f'{transcript}\n')

    spm.SentencePieceTrainer.Train(
        f'--input={datapath}/sentencepiece_input.txt '
        '--model_prefix=kspon_sentencepiece '
        f'--vocab_size={vocab_size} '
        '--model_type=bpe '
        '--max_sentence_length=9999 '
        '--hard_vocab_limit=false '
        '--pad_id=0 '
        '--bos_id=1 '
        '--eos_id=2 '
        '--unk_id=3 '
    )


def do_subword_process(transcript: str, instance: spm.SentencePieceProcessor):
    pieces = instance.EncodeAsPieces(transcript)
    ids = instance.PieceToId(pieces)
    return pieces, ids


def sentence_to_subwords(audio_paths: list, transcripts: list, datapath: str = './data',
                         batch_step: int = 256):
    subwords = list()

    print('sentence_to_subwords...')

    sp = spm.SentencePieceProcessor()
    vocab_file = "kspon_sentencepiece.model"
    sp.load(vocab_file)

    with Parallel(n_jobs=cpu_count() - 1) as parallel:
        with open(f'{datapath}/transcripts.txt', 'w') as f:
            for batch_idx in tqdm(range(0, len(audio_paths), batch_step), desc='Subword Process...'):
                audio_paths_subset = audio_paths[batch_idx * batch_step: (batch_idx + 1) * batch_step]
                transcripts_subset = transcripts[batch_idx * batch_step: (batch_idx + 1) * batch_step]

                results_subset = parallel(
                    delayed(do_subword_process)(transcript, sp) for transcript in transcripts_subset
                )

                subset_results = []
                for audio_path, (subword_transcript, subword_id) in zip(audio_paths_subset, results_subset):
                    subword_transcript = ' '.join(subword_transcript)
                    subword_id = ' '.join(list(map(str, subword_id)))
                    subset_results.append(f'{audio_path}\t{subword_transcript}\t{subword_id}')

                f.write('\n'.join(subset_results)+'\n')

    return subwords
