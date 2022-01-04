# input image size
WIDTH = 96
HEIGHT = 24
CHANNEL = 1

SEED = 1234
N_SAMPLE = 1000
# training config
BATCH_SIZE = 256
# TRAIN_SAMPLE = 5295
TRAIN_SAMPLE = 104916
NUM_EPOCHS = 200
WARMUP_EPOCH = 10
# LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4

CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopq"  # exclude IO
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}

# output label length
NUM_CLASS = len(CHARS)+1
LABEL_MAX_LEN = 8

TRAIN_DIR = r'C:\dataset\license_plate\license_plate_recognition\single\train'
VAL_DIR = r'C:\dataset\license_plate\license_plate_recognition\single\val'
# TRAIN_DIR = r'C:\dataset\license_plate\mini_LPR_dataset\train'
# VAL_DIR = r'C:\dataset\license_plate\mini_LPR_dataset\val'

TEST_DIR = VAL_DIR
# TEST_DIR = r'C:\dataset\license_plate\license_plate_recognition\single\val'
