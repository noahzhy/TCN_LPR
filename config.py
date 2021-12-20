# input image size
WIDTH = 96
HEIGHT = 64
CHANNEL = 1

# NUM_CLASSES = 86

# training config
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 5e-3

CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ"  # exclude IO
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}
# output label length
NUM_CLASS = len(CHARS)+1
