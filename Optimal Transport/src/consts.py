from pathlib import Path

DATA_P = Path("data")
REUTERS_P = DATA_P / "reuters"
GLOVE_P = DATA_P / Path("embeddings") / "glove"
GLOVE_50D = "glove.6B.50d.txt"
GLOVE_F = GLOVE_P / GLOVE_50D

TFIDF_P = DATA_P / "X.npz"
VOC_P = DATA_P / "voc.pkl"
GLOVE_REUTERS_P = DATA_P / "glove_reuters.pkl"
MODEL_SAVE_P = DATA_P / "model"
MODEL_SAVE_SUFFIX = ".pkl"
TRAIN_TEST_P = DATA_P / "train_test"
RC20_TOPICS = 20

EMBED_DIM = 50 # dimention of words and topics embeddings
HIDDEN_DIM = 100 # hidden dimention of the encoder
DROPOUT_R = 0.75
BATCH_SIZE = 100 # batch size = 200 in original paper.
TOPICS_SIZE = RC20_TOPICS

LEARNING_RATE = 0.01
BETA = .8

EPSILON = 0.007 #  wheight of the cross-entropy term of the loss
ALPHA = 20 # entropic regularizer
MAX_ITER = 100 # max OT iters
STOP_TOLERANCE = 0.01 # OT tolerance paper : 0.005

# weirdly enough the authors did a maximum of 50 batchs of dimentions 200.
# given the difficulties I had with NaNs, I believe they might have stopped early for the same
# reason.
MAX_OUTER_ITER = 50

# For the reuters dataset, 50 batchs of 200 ex is ~ 1 epoch.
N_EPOCHS = 3


