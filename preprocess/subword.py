import os
import sentencepiece as spm


def train_sentencepiece(transcripts, datapath: str = './data', vocab_size: int = 5000):
    print('generate_sentencepiece_input..')

    if not os.path.exists(datapath):
        os.mkdir(datapath)

    with open(f'{datapath}/sentencepiece_input.txt', 'w', encoding="utf-8") as f:
        for transcript in transcripts:
            transcript = transcript.upper()
            f.write(f'{transcript}\n')

    spm.SentencePieceTrainer.Train(
        f"--input={datapath}/sentencepiece_input.txt "
        f"--model_prefix=sp "
        f'--vocab_size={vocab_size} '
        f"--model_type={SENTENCEPIECE_MODEL_TYPE} "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
        f"--user_defined_symbols={blank_token}"
    )


def convert_subword(transcript: str, sp: spm.SentencePieceProcessor):
    text = " ".join(sp.EncodeAsPieces(transcript))
    label = " ".join([str(sp.PieceToId(token)) for token in text])
    return text, label


def sentence_to_subwords(
        audio_paths: list,
        transcripts: list,
        manifest_file_path: str,
) -> None:
    sp = spm.SentencePieceProcessor()
    sp.Load("sp.model")

    with open(manifest_file_path, 'w', encoding="utf-8") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            text, label = convert_subword(transcript, sp)
            f.write(f"{audio_path}\t{text}\t{label}\n")
