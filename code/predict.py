import warnings

warnings.filterwarnings("ignore")

import torch

import os
import argparse
import numpy as np
import json
from tqdm import tqdm

from ark_nlp.factory.utils.seed import set_seed
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from models.global_pointer_nezha import GlobalPointerNeZha
from models.configuration_nezha import NeZhaConfig
from gpPredictor import GlobalPointerNERPredictor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_args():
    parser = argparse.ArgumentParser(description="ner")
    parser.add_argument("--test_data_path", default=os.path.join(BASE_DIR, "data/contest_data/preliminary_test_a"),
                        type=str, help="test data file path")
    parser.add_argument("--test_file_name", default="sample_per_line_preliminary_A",
                        type=str, help="predict file name")
    parser.add_argument("--test_data_file_path", default="", type=str)
    parser.add_argument("--best_model_name", default=os.path.join(BASE_DIR, "data/best_model"), type=str)
    parser.add_argument("--result_save_path", default=os.path.join(BASE_DIR, "submission"),
                        type=str, help="result file path")
    parser.add_argument("--result_name", default="result.txt", type=str)

    parser.add_argument("--cuda_device", default=4, type=int, help="the number of cuda to use")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    return args


args=get_args()
set_seed(args.seed)


def predict(model, tokenizer, cat2id):
    ner_predictor_instance = GlobalPointerNERPredictor(model, tokenizer, cat2id)

    predict_results = []

    if args.test_data_file_path and os.path.isfile(args.test_data_file_path):
        test_data_file_path = args.test_data_file_path
    else:
        test_data_file_path = os.path.join(args.test_data_path, args.test_file_name)

    with open(test_data_file_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            label = len(_line) * ['O']
            for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
                if 'I' in label[_preditc['start_idx']]:
                    continue
                if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                    continue
                if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                    continue

                label[_preditc['start_idx']] = 'B-' + _preditc['type']
                label[_preditc['start_idx'] + 1: _preditc['end_idx'] + 1] = (_preditc['end_idx'] - _preditc[
                    'start_idx']) * [('I-' + _preditc['type'])]

            predict_results.append([_line, label])
    # with open("pre_train_unlabeled_data_100000.txt", "wt", encoding="utf-8") as f:
    #     f.write(json.dumps(predict_results, ensure_ascii=False))
    return predict_results


def save_result(predict_results, path_dir="./"):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(os.path.join(path_dir, args.result_name), 'wt', encoding='utf-8') as f:
        for _result in predict_results:
            for word, tag in zip(_result[0], _result[1]):
                if word == '\n':
                    continue
                f.write(f'{word} {tag}\n')
            f.write('\n')


def get_label_dict():
    cat2id_list = sorted([str(i) for i in range(1, 55) if i != 27 and i != 45] + ["O"])
    return {key:idx for idx, key in enumerate(cat2id_list)}


if __name__ == "__main__":

    tokenizer = Tokenizer(vocab=args.best_model_name, max_seq_len=128)
    cat2id = get_label_dict()
    config = NeZhaConfig.from_pretrained(args.best_model_name, num_labels=len(cat2id))
    model = GlobalPointerNeZha.from_pretrained(args.best_model_name,
                                               config=config)
    device = torch.device("cuda:{}".format(args.cuda_device) if torch.cuda.is_available() else "cpu")
    model.to(device)
    predict_results = predict(model, tokenizer, cat2id)
    save_result(predict_results, args.result_save_path)


