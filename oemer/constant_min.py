from oemer.dense_dataset_definitions import DENSE_DATASET_DEFINITIONS as DEF

CLASS_CHANNEL_LIST = [
    DEF.STAFF + DEF.LEDGERLINE,
    DEF.NOTEHEADS_ALL + DEF.STEM + [38, 42, 46],
    (DEF.FLAG_UP + DEF.FLAG_DOWN + DEF.BEAM + [65, 59] # flags, beam
     + DEF.ALL_RESTS_EXCEPT_LARGE + [163] # rests
     + DEF.ALL_ACCIDENTALS + DEF.ALL_KEYS + DEF.BARLINE_BETWEEN
     + DEF.ALL_CLEFS + DEF.NUMBERS + DEF.DOT 
     + DEF.TIME_SIGNATURE_SUBSET),
]

CLASS_CHANNEL_MAP = {
    color: idx+1
    for idx, colors in enumerate(CLASS_CHANNEL_LIST)
    for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 2
