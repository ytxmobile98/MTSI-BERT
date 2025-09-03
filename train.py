import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertTokenizer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import (KvretConfig, KvretDataset, MTSIAdapterDataset, MTSIBert,
                   MTSIKvretConfig, TwoSepTensorBuilder)

_N_EPOCHS = 20
# how many samples has to be computed before the optimizer.step()
_OPTIMIZER_STEP_RATE = 16
_DEBUG = False


def get_eos(turns, win_size, windows_per_dialogue):

    res = torch.zeros((len(turns), windows_per_dialogue), dtype=torch.long)
    user_count = 0
    for idx, curr_dial in enumerate(turns):
        for t in curr_dial:
            if t == 1:
                user_count += 1
        res[idx][user_count-1] = 1

    return res, user_count-1


def remove_dataparallel(load_checkpoint_path):
    # original saved file with DataParallel
    state_dict = torch.load(load_checkpoint_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict


def train(load_checkpoint_path=None):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    if not _DEBUG:
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        device = 'cpu'
    print('active device = '+str(device))

    # Dataset preparation
    training_set = KvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    training_set.remove_subsequent_actor_utterances()
    validation_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)
    validation_set.remove_subsequent_actor_utterances()

    # Bert adapter for dataset
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=False)
    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS) + 1 (random last sentence from other)
    badapter_train = MTSIAdapterDataset(training_set, tokenizer,
                                        KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,
                                        KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    # for validation keep using the train max tokens for model compatibility
    badapter_val = MTSIAdapterDataset(validation_set, tokenizer,
                                      KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,
                                      KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_VAL_DIALOGUE+2)

    # Parameters
    params = {'batch_size': MTSIKvretConfig._BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = MTSIBert(num_layers_encoder=MTSIKvretConfig._ENCODER_LAYERS_NUM,
                     num_layers_eos=MTSIKvretConfig._EOS_LAYERS_NUM,
                     n_intents=MTSIKvretConfig._N_INTENTS,
                     batch_size=MTSIKvretConfig._BATCH_SIZE,
                     pretrained='bert-base-cased',
                     seed=MTSIKvretConfig._SEED,
                     window_size=MTSIKvretConfig._WINDOW_SIZE)

    if load_checkpoint_path is not None:
        print('model loaded from: '+load_checkpoint_path)
        new_state_dict = remove_dataparallel(load_checkpoint_path)
        model.load_state_dict(new_state_dict)
    # work on multiple GPUs when available
    if torch.cuda.device_count() > 1:
        print('active devices = '+str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # these weights are needed because of unbalancing between 0 and 1 for action and eos
    loss_eos_weights = torch.tensor([1, 2.6525])
    loss_action_weights = torch.tensor([1, 4.8716])
    loss_eos = torch.nn.CrossEntropyLoss(weight=loss_eos_weights).to(device)
    loss_action = torch.nn.CrossEntropyLoss(
        weight=loss_action_weights).to(device)
    loss_intent = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model._bert.parameters(
            ), "lr": MTSIKvretConfig._BERT_LEARNING_RATE},
            {"params": model._encoderbiLSTM.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
            {"params": model._eos_ffnn.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
            {"params": model._intent_ffnn.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
            {"params": model._action_ffnn.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
            {"params": model._eos_classifier.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
            {"params": model._intent_classifier.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
            {"params": model._action_classifier.parameters(
            ), "lr": MTSIKvretConfig._NN_LEARNING_RATE},
        ],
        weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 10, 15, 20, 30, 40, 50, 75], gamma=0.5)

    # creates the directory for the checkpoints
    os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH), exist_ok=True)
    curr_date = datetime.datetime.now().isoformat()
    os.makedirs((MTSIKvretConfig._SAVING_PATH /
                 curr_date).parent, exist_ok=True)
    # creates the directory for the plots figure
    os.makedirs(
        MTSIKvretConfig._PLOTS_SAVING_PATH.parent, exist_ok=True)

    # initializes the losses lists
    best_loss = 100
    train_global_losses = []
    val_global_losses = []
    eos_val_global_losses = []
    action_val_global_losses = []
    intent_val_global_losses = []

    # ------------- TRAINING -------------

    logging.basicConfig(level=logging.INFO)

    tensor_builder = TwoSepTensorBuilder()

    for epoch in tqdm(range(_N_EPOCHS), desc="Epoch"):

        model.train()
        t_eos_losses = []
        t_intent_losses = []
        t_action_losses = []

        for curr_step, (
            local_batch, local_turns, local_intents,
            local_actions, dialogue_ids) in tqdm(
                enumerate(training_generator),
                total=len(training_generator),
                desc='Batch step'):

            # 0 = intra dialogue ; 1 = eos
            eos_label, eos_idx = get_eos(local_turns, MTSIKvretConfig._WINDOW_SIZE,
                                         windows_per_dialogue=KvretConfig._KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE + 2)

            # local_batch.shape == B x D_LEN x U_LEN
            # local_intents.shape == B
            # local_actions.shape == B
            # local_eos_label.shape == B x D_PER_WIN
            local_batch = local_batch.to(device)
            local_intents = local_intents.to(device)
            local_actions = local_actions.to(device)
            eos_label = eos_label.to(device)

            eos, intent, action = model(
                local_batch, local_turns, dialogue_ids, tensor_builder, device)

            # compute loss only on real dialogue (exclude padding)
            loss1 = loss_eos(eos['logit'][:eos_idx+1],
                             eos_label.squeeze(0)[:eos_idx+1])
            loss2 = loss_intent(intent['logit'], local_intents)
            loss3 = loss_action(action['logit'], local_actions)
            tot_loss = (loss1 + loss2 + loss3)/3
            tot_loss.backward()

            # save results
            t_eos_losses.append(loss1.item())
            t_intent_losses.append(loss2.item())
            t_action_losses.append(loss3.item())

            if curr_step != 0 and curr_step % _OPTIMIZER_STEP_RATE == 0 or curr_step == badapter_train.__len__()-1:
                optimizer.step()
                optimizer.zero_grad()

            if 'cuda' in str(device):
                torch.cuda.empty_cache()

        # end of epoch

        # ------------- VALIDATION -------------
        with torch.no_grad():
            model.eval()
            v_eos_losses = []
            v_intent_losses = []
            v_action_losses = []

            for local_batch, local_turns, local_intents, local_actions, dialogue_ids in validation_generator:

                # 0 = intra dialogue ; 1 = eos
                eos_label, eos_idx = get_eos(local_turns, MTSIKvretConfig._WINDOW_SIZE,
                                             windows_per_dialogue=KvretConfig._KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE + 1)

                # local_batch.shape == B x D_LEN x U_LEN
                # local_intents.shape == B
                # local_actions.shape == B
                # local_eos_label.shape == B x D_PER_WIN
                local_batch = local_batch.to(device)
                local_intents = local_intents.to(device)
                local_actions = local_actions.to(device)
                eos_label = eos_label.to(device)

                eos, intent, action = model(
                    local_batch, local_turns, dialogue_ids,
                    tensor_builder, device)

                if 'cuda' in str(device):
                    torch.cuda.empty_cache()

                loss1 = loss_eos(eos['logit'][:eos_idx+1],
                                 eos_label.squeeze(0)[:eos_idx+1])
                loss2 = loss_intent(intent['logit'], local_intents)
                loss3 = loss_action(action['logit'], local_actions)

                # save results
                v_eos_losses.append(loss1.item())
                v_intent_losses.append(loss2.item())
                v_action_losses.append(loss3.item())

        # compute the mean for each loss in the current epoch
        t_eos_curr_mean = round(np.mean(t_eos_losses), 4)
        t_action_curr_mean = round(np.mean(t_action_losses), 4)
        t_intent_curr_mean = round(np.mean(t_intent_losses), 4)

        v_eos_curr_mean = round(np.mean(v_eos_losses), 4)
        v_action_curr_mean = round(np.mean(v_action_losses), 4)
        v_intent_curr_mean = round(np.mean(v_intent_losses), 4)

        train_mean_loss = round(
            np.mean([t_eos_curr_mean, t_action_curr_mean, t_intent_curr_mean]), 4)
        val_mean_loss = round(
            np.mean([v_eos_curr_mean, v_action_curr_mean, v_intent_curr_mean]), 4)

        # accumulate losses for plotting
        eos_val_global_losses.append(v_eos_curr_mean)
        action_val_global_losses.append(v_action_curr_mean)
        intent_val_global_losses.append(v_intent_curr_mean)
        train_global_losses.append(train_mean_loss)
        val_global_losses.append(val_mean_loss)

        # check if new best model
        if val_mean_loss < best_loss:
            # saves the model weights
            best_loss = val_mean_loss
            # save using model.cpu to allow the further loading
            # also on cpu or single-GPU
            save_dir = MTSIKvretConfig._SAVING_PATH / curr_date
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.cpu().state_dict(), save_dir / 'state_dict.pt')
        model.to(device)

        bert_curr_lr = optimizer.param_groups[0]['lr']
        nn_curr_lr = optimizer.param_groups[1]['lr']
        log_str = f'### EPOCH {epoch+1}/{_N_EPOCHS} (bert_lr={bert_curr_lr},' \
            f' nn_lr={nn_curr_lr}):: TRAIN LOSS = {train_mean_loss} ' \
            f' [eos = {round(np.mean(t_eos_losses), 4)}],' \
            f' [action = {round(np.mean(t_action_losses), 4)}],' \
            f' [intent = {round(np.mean(t_intent_losses), 4)}],' \
            f'\n\t\t\t || VAL LOSS = {val_mean_loss}' \
            f' [eos = {round(np.mean(v_eos_losses), 4)}],' \
            f' [action = {round(np.mean(v_action_losses), 4)}],' \
            f' [intent = {round(np.mean(v_intent_losses), 4)}]'
        print(log_str)
        # step of scheduler to reduce the lr each milestone
        scheduler.step()

    # ------------ FINAL PLOTS ------------

    epoch_list = np.arange(1, _N_EPOCHS+1)

    os.makedirs(MTSIKvretConfig._PLOTS_SAVING_PATH, exist_ok=True)

    # plot train vs validation
    plt.plot(epoch_list, train_global_losses, color='blue', label='train loss')
    plt.plot(epoch_list, val_global_losses,
             color='red', label='validation loss')

    plt.title('train vs validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(MTSIKvretConfig._PLOTS_SAVING_PATH / 'train_vs_val.png')

    # clean figure
    plt.clf()

    # plot eos vs action vs intent
    plt.plot(epoch_list, eos_val_global_losses, color='red', label='eos loss')
    plt.plot(epoch_list, action_val_global_losses,
             color='green', label='action loss')
    plt.plot(epoch_list, intent_val_global_losses,
             color='blue', label='intent loss')

    plt.title('eos vs action vs intent')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend(loc='best')
    plt.savefig(MTSIKvretConfig._PLOTS_SAVING_PATH / 'validation_losses.png')


if __name__ == '__main__':
    start = time.time()
    with torch.autograd.set_detect_anomaly(True):
        train()
    end = time.time()
    h_count = (end-start)/60/60
    print('training time: '+str(h_count)+'h')
