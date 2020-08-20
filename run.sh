# Author
# Soohwan Kim, Seyoung Bae

DATASET_PATH="SET_YOUR_DATASET_PATH"
NEW_PATH="SET_YOUR_PATH_TO_SAVE_SCRIPT"
SCRIPT_PREFIX='KsponScript_'
LABELS_DEST='SET_LABELS_DESTINATION'
GRAPHEME_SAVE_PATH='SET_YOUR_PATH_TO_SAVE_GRAPHME_TEXT'
OUTPUT_UNIT='character'       # you can set character / subword / grapheme
PREPROCESS_MODE='phonetic'    # phonetic : 칠 십 퍼센트,  spelling : 70%
VOCAB_SIZE=5000               # if you use subword output unit, set vocab size


python main.py --dataset_path "$DATASET_PATH" --new_path "$NEW_PATH" --script_prefix $SCRIPT_PREFIX \
--labels_dest $LABELS_DEST --output_unit $OUTPUT_UNIT --preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE --grapheme_save_path $GRAPHEME_SAVE_PATH
