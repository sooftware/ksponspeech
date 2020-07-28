DATASET_PATH="SET YOUR KsponSpeech corpus PATH"
NEW_PATH="SET YOUR path to store preprocessed KsponSpeech corpus"
SCRIPT_PREFIX='KsponScript_'
MODE = 'CHOOSE : phonetic or numeric'
FILENUM_ADJUST = 'set True to handle "%" if some files are deleted; you must know how many files are gone and therefore how much these files should drift.'

python main.py --dataset_path "$DATASET_PATH" --new_path "$NEW_PATH" --script_prefix $SCRIPT_PREFIX --mode "$MODE" --filenum_adjust "$FILENUM_FILENUM_ADJUST"
