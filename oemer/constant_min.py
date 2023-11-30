from oemer.dense_dataset_definitions import DENSE_DATASET_DEFINITIONS as DEF


CLASS_CHANNEL_LIST = [
    DEF.BARLINE_END + DEF.BARLINE_BETWEEN,
    DEF.NOTEHEADS_ALL,
    DEF.ALL_CLEFS,
]

CLASS_CHANNEL_MAP = {
    color: idx+1
    for idx, colors in enumerate(CLASS_CHANNEL_LIST)
    for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 2
