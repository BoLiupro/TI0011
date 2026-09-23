import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'beijing')
SAVE_DIR = os.path.join(BASE_DIR, 'save')

LOCATION_FILE = os.path.join(DATA_DIR, 'location.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'mobility_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'mobility_test.csv')

# geographic bounds of the study area (Beijing subregion) -- must match SeqGAN/beijing/config.py
LON_MIN = 116.25
LON_MAX = 116.50
LAT_MIN = 39.85
LAT_MAX = 40.05
GRID_ROWS = 20
GRID_COLS = 25

# sequence / vocab
SEQ_LEN = 24          # hourly slots per day
NUM_REGIONS = 500     # grid_rows * grid_cols
VOCAB_SIZE = NUM_REGIONS  # MoveSim never predicts position 0, so no BOS token is needed

# generator (attention-gated Markov transition model, ATGenerator in the reference code)
LOC_EMB_DIM = 64
TIM_EMB_DIM = 16
HIDDEN_DIM = 64

# discriminator (TextCNN) hyperparameters -- filter config from the MoveSim paper's own code
DIS_EMB_DIM = 64
DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
DIS_NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
DIS_DROPOUT_KEEP_PROB = 0.75

# training
BATCH_SIZE = 256      # bigger than SeqGAN's 128: MoveSim's per-step compute is much cheaper
SEED = 88

PRE_EPOCH_G = 40       # MLE pretraining epochs for the generator
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
DLOSS_ALPHA = 1.5           # weight of the auxiliary consecutive-step distance regularizer

GENERATOR_PRETRAIN_CKPT = os.path.join(SAVE_DIR, 'generator_pretrain.pt')
GENERATOR_CKPT = os.path.join(SAVE_DIR, 'generator.pt')
DISCRIMINATOR_CKPT = os.path.join(SAVE_DIR, 'discriminator.pt')
TRAIN_LOG = os.path.join(SAVE_DIR, 'train_log.txt')
EVAL_RESULTS = os.path.join(SAVE_DIR, 'eval_results.txt')
COMPARE_RESULTS = os.path.join(SAVE_DIR, 'compare_next_location.txt')

DEVICE = 'cuda'
