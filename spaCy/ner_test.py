import argparse
import pathlib
import spacy
from spacy.language import Language
from spacy.training import Example
from ner_kvret import NERKvretDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from MTSIBertConfig import KvretConfig


def parse_args():
    parser = argparse.ArgumentParser(description="spaCy NER model testing")
    parser.add_argument("--model-dir", type=pathlib.Path, required=True,
                        help="Directory of the trained spaCy model")
    return parser.parse_args()


def spacy_test(spacy_model: Language, data):

    doc = spacy_model('')

    missing_entities = 0
    bleu_scores = []

    smoothing_function = SmoothingFunction().method1

    examples: list[Example] = []

    for t_sample in data:
        utt = t_sample[0]
        ents_l = t_sample[1]['entities']
        ents_l.sort(key=lambda tup: tup[0])  # sorts in place the entities list

        doc = spacy_model(utt)  # make prediction
        if len(doc.ents) > len(ents_l):
            missing_entities += (len(doc.ents) - len(ents_l))

        for pred, truth in zip(doc.ents, ents_l):
            start_idx = truth[0]
            end_idx = truth[1]
            curr_bleu = sentence_bleu(
                references=utt[start_idx:end_idx],
                hypothesis=str(pred),
                weights=(1.0, 0.0, 0.0, 0.0),
                smoothing_function=smoothing_function)
            bleu_scores.append(curr_bleu)

        examples.append(Example.from_dict(doc, t_sample[1]))

    eval_results = spacy_model.evaluate(examples, per_component=True)
    print("Evaluation results:", eval_results)
    print('BLEU: '+str(np.mean(bleu_scores)))
    print('missing: '+str(missing_entities))


if __name__ == '__main__':
    args = parse_args()

    test_set = NERKvretDataset(KvretConfig._KVRET_TEST_PATH)
    stest_set = test_set.build_spacy_dataset()
    spacy_model_dir = args.model_dir
    model = spacy.load(spacy_model_dir)

    spacy_test(model, stest_set)
