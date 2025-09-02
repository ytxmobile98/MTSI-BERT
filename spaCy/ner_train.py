import os
import datetime
import time
import random
import spacy
from spacy.training import Example
from ner_kvret import NERKvretDataset, Entities
from MTSIBertConfig import KvretConfig, PROJECT_ROOT_DIR

_N_EPOCHS = 120
_BATCH_SIZE = 32

_SPACY_MODEL_SAVING_PATH = PROJECT_ROOT_DIR / 'spaCy_savings'


def spacy_train(data: list[tuple[str, dict[str, Entities]]]) \
        -> spacy.language.Language:
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.initialize()

        for epoch in range(_N_EPOCHS):
            random.shuffle(straining_set)
            losses = {}

            for idx in range(0, len(data), _BATCH_SIZE):
                curr_batch_items = data[idx:idx+_BATCH_SIZE]

                examples = [
                    Example.from_dict(
                        nlp.make_doc(text), annotations)
                    for text, annotations in curr_batch_items
                ]
                nlp.update(
                    examples=examples,
                    drop=0.2,  # dropout
                    sgd=optimizer,  # callable to update weights
                    losses=losses)

            log_str = f'### EPOCH {epoch+1}/{_N_EPOCHS}:: TRAIN LOSS = {losses}'
            print(log_str)

    return nlp


if __name__ == '__main__':
    start = time.time()
    curr_date = datetime.datetime.now().isoformat()

    training_set = NERKvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    straining_set = training_set.build_spacy_dataset()

    validation_set = NERKvretDataset(KvretConfig._KVRET_VAL_PATH)
    svalidation_set = validation_set.build_spacy_dataset()

    # train on both training and validation set
    spacy_model = spacy_train(straining_set+svalidation_set)
    save_dir = _SPACY_MODEL_SAVING_PATH / 'ner' / curr_date
    os.makedirs(save_dir, exist_ok=True)
    spacy_model.to_disk(save_dir)

    end = time.time()
    h_count = (end-start)/60/60
    print(f'training time: {h_count}h')
    print(f'Model saved to: "{save_dir.absolute()}"')
