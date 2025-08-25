import os
import datetime
import sys
import time
import random
import spacy
from spacy.training import Example
from ner_kvret import NERKvretDataset

sys.path.insert(1, 'model/')
from MTSIBertConfig import KvretConfig


_N_EPOCHS = 120
_SPACY_MODEL_SAVING_PATH = 'spaCy_savings/'
_BATCH_SIZE = 32


def spacy_train(data):
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        nlp.add_pipe('ner')
    ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()

        for epoch in range(_N_EPOCHS):
            random.shuffle(straining_set)
            losses = {}
            curr_batch_text = []
            curr_batch_label = []
            for idx, (text, annotations) in enumerate(data):
                if idx % _BATCH_SIZE == 0 or idx == len(data) - 1:
                    curr_batch_text.append(text)
                    curr_batch_label.append(annotations)
                examples = [
                    Example.from_dict(
                        nlp.make_doc(text), annotations)
                    for text, annotations in zip(
                        curr_batch_text, curr_batch_label)
                ]
                nlp.update(
                    examples=examples,
                    drop=0.2,  # dropout
                    sgd=optimizer,  # callable to update weights
                    losses=losses)

                curr_batch_text = []
                curr_batch_label = []

            log_str = '### EPOCH '+str(epoch+1)+'/'+str(_N_EPOCHS)+':: TRAIN LOSS = '+str(losses)
            print(log_str)

    return nlp


if __name__ == '__main__':
    start = time.time()
    curr_date = datetime.datetime.now().isoformat()
    # creates the directory for the checkpoints
    os.makedirs(os.path.dirname(_SPACY_MODEL_SAVING_PATH), exist_ok=True)

    training_set = NERKvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    straining_set = training_set.build_spacy_dataset()

    validation_set = NERKvretDataset(KvretConfig._KVRET_VAL_PATH)
    svalidation_set = validation_set.build_spacy_dataset()

    # train on both training and validation set
    spacy_model = spacy_train(straining_set+svalidation_set)
    save_dir = os.path.join(_SPACY_MODEL_SAVING_PATH, 'ner')
    os.makedirs(save_dir, exist_ok=True)
    spacy_model.to_disk(os.path.join(save_dir, curr_date))

    end = time.time()
    h_count = (end-start)/60/60
    print('training time: '+str(h_count)+'h')
