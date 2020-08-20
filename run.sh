DATASET_PATH="E:/KsponSpeech/original"
NEW_PATH="E:/KsponSpeech/character"
SCRIPT_PREFIX='KsponScript_'
LABELS_DEST='E:/KsponSpeech'


python main.py --dataset_path "$DATASET_PATH" --new_path "$NEW_PATH" --script_prefix $SCRIPT_PREFIX \
--labels_dest $LABELS_DEST