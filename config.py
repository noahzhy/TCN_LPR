# input image size
WIDTH = 96
HEIGHT = 32
CHANNEL = 1

SEED = 1234
N_SAMPLE = 2000
# training config
BATCH_SIZE = 256
NUM_EPOCHS = 300
WARMUP_EPOCH = 5
# LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4

CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ"  # exclude IO
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}

# output label length
NUM_CLASS = len(CHARS)+1
LABEL_MAX_LEN = 8

TRAIN_DIR = r'C:\dataset\license_plate\license_plate_recognition\train'
VAL_DIR = r'C:\dataset\license_plate\license_plate_recognition\val'
# TRAIN_DIR = r'images'
# VAL_DIR = r'images'
TEST_DIR = VAL_DIR
