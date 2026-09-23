import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'app_usage', 'shanghai', 'derived')
SAVE_DIR = os.path.join(BASE_DIR, 'save')

LOCATION_FILE = os.path.join(DATA_DIR, 'location.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'mobility_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'mobility_test.csv')

# geographic bounds of the study area (1st-99th percentile bbox of app_usage/shanghai/location.csv)
LON_MIN = 121.03329658749999
LON_MAX = 121.82629967
LAT_MIN = 30.73458719
LAT_MAX = 31.433447360000002
GRID_ROWS = 20
GRID_COLS = 20

# sequence / vocab
SEQ_LEN = 24          # hourly slots per day
NUM_REGIONS = 400     # grid_rows * grid_cols
BOS = NUM_REGIONS     # dedicated start-of-sequence id, embedding-only, never predicted
VOCAB_SIZE = NUM_REGIONS  # output head size (BOS excluded)

# generator (GPT-style transformer) hyperparameters
EMB_DIM = 64
N_LAYER = 4
N_HEAD = 4
FF_DIM = 256
DROPOUT = 0.1

# discriminator (TextCNN) hyperparameters
DIS_EMB_DIM = 64
DIS_FILTER_SIZES = [2, 3, 4, 5, 6, 8, 10, 12]
DIS_NUM_FILTERS = [64, 64, 64, 64, 64, 64, 64, 64]
DIS_DROPOUT_KEEP_PROB = 0.75
DIS_L2_REG_LAMBDA = 0.2

# training
BATCH_SIZE = 128
SEED = 88

PRE_EPOCH_G = 30          # MLE pretraining epochs for the generator
G_LR = 1e-3

GENERATED_NUM = 5000       # size of the generated (fake) pool used for discriminator training
D_PRETRAIN_ROUNDS = 20     # rounds of (generate negatives, train D a few epochs)
D_PRETRAIN_EPOCHS_PER_ROUND = 3
D_LR = 1e-4

ADV_TOTAL_ROUNDS = 100      # adversarial training rounds
ADV_ROLLOUT_NUM = 8
ADV_D_REFRESH_EVERY = 5     # refresh discriminator every N generator updates
ADV_D_ROUNDS = 1            # (generate negatives, train D) repeats per refresh
ADV_D_EPOCHS_PER_ROUND = 1
ADV_G_LR = 1e-4
REWARD_GAMMA = 0.95

GENERATOR_PRETRAIN_CKPT = os.path.join(SAVE_DIR, 'generator_pretrain.pt')
GENERATOR_CKPT = os.path.join(SAVE_DIR, 'generator.pt')
DISCRIMINATOR_CKPT = os.path.join(SAVE_DIR, 'discriminator.pt')
TRAIN_LOG = os.path.join(SAVE_DIR, 'train_log.txt')
EVAL_RESULTS = os.path.join(SAVE_DIR, 'eval_results.txt')

# mobility generation eval
GEN_NUM_SAMPLES = 20   # K stochastic continuations per test sequence

DEVICE = 'cuda'
