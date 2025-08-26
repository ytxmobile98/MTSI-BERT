import os
import sys
import spacy
from ner_kvret import NERKvretDataset
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

sys.path.insert(1, 'model/')
from MTSIBertConfig import KvretConfig

_SPACY_MODEL_SAVING_PATH = 'spaCy_savings/ner'


def spacy_test(spacy_model, data):

    doc = spacy_model('Where is the nearest gas station?')

    missing_entities = 0
    bleu_scores = []

    for t_sample in data:
        utt = t_sample[0]
        ents_l = t_sample[1]['entities']
        ents_l.sort(key=lambda tup: tup[0])  # sorts in place the entities list

        doc = spacy_model(utt)  # make prediction

        #if len(doc.ents) > len(ents_l):
            #pdb.set_trace()
        #assert len(ents_l) >= len(doc.ents), 'PREDICTED MORE ENTITIES THAN REQUESTED'
        if len(doc.ents) > len(ents_l):
            missing_entities += (len(doc.ents) - len(ents_l))

        for pred, truth in zip(doc.ents, ents_l):
            start_idx = truth[0]
            end_idx = truth[1]
            curr_bleu = sentence_bleu(references=utt[start_idx:end_idx], hypothesis=pred)
            bleu_scores.append(curr_bleu)

    print('BLEU: '+str(np.mean(bleu_scores)))
    print('missing: '+str(missing_entities))


if __name__ == '__main__':
    test_set = NERKvretDataset(KvretConfig._KVRET_TEST_PATH)
    stest_set = test_set.build_spacy_dataset()
    # spacy_model_path = 'spaCy/spaCy_savings/spacy_2019-08-25T22:23:47.579104'
    spacy_model_path = os.path.join(
        _SPACY_MODEL_SAVING_PATH, '2025-08-26T09:32:52.249503')
    model = spacy.load(spacy_model_path)

    spacy_test(model, stest_set)
