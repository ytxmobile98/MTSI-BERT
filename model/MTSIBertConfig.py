from pathlib import Path

"""
A module containing all the infos for the project
"""

PROJECT_ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT_DIR / 'dataset' / 'kvret_dataset_public'
SAVINGS_DIR = PROJECT_ROOT_DIR / 'savings'
PLOTS_DIR = PROJECT_ROOT_DIR / 'plots'


class KvretConfig():
    """
    Kvret Dataset configuration parameters
    """
    _KVRET_TRAIN_PATH = DATASET_DIR / 'kvret_train_public.json'
    _KVRET_VAL_PATH = DATASET_DIR / 'kvret_dev_public.json'
    _KVRET_TEST_PATH = DATASET_DIR / 'kvret_test_public.json'
    _KVRET_ENTITIES_PATH = DATASET_DIR / 'kvret_entities.json'

    _KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE = 113
    _KVRET_MAX_BERT_TOKENS_PER_VAL_SENTENCE = 111
    _KVRET_MAX_BERT_TOKENS_PER_TEST_SENTENCE = 50

    _KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE = 12
    _KVRET_MAX_BERT_SENTENCES_PER_VAL_DIALOGUE = 12
    _KVRET_MAX_BERT_SENTENCES_PER_TEST_DIALOGUE = 12

    _KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE = 6
    _KVRET_MAX_USER_SENTENCES_PER_VALIDATION_DIALOGUE = 6
    # 7 if subsequent utterances not removed
    _KVRET_MAX_USER_SENTENCES_PER_TEST_DIALOGUE = 6

    _KVRET_MAX_BERT_TOKENS_PER_WINDOWS = 129  # see kvret statistics


class MTSIKvretConfig:
    """
    MTSI-Bert model parameters for Kvret dataset
    """
    _N_INTENTS = 3  # number of intents
    _BATCH_SIZE = 1
    _ENCODER_LAYERS_NUM = 1
    _EOS_LAYERS_NUM = 1
    _SEED = 26  # for reproducibility of results
    _BERT_LEARNING_RATE = 5e-5
    _NN_LEARNING_RATE = 1e-3
    _WINDOW_SIZE = 3  # tipically odd number [Q(t-1), R(t-1), Q(t)]

    _SAVING_PATH = SAVINGS_DIR
    _PLOTS_SAVING_PATH = PLOTS_DIR
