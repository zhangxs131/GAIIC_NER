import warnings

warnings.filterwarnings("ignore")

import torch
from torchcontrib.optim import SWA
import pandas as pd
import os
import time
import json

from ark_nlp.factory.utils.seed import set_seed
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from models.global_pointer_nezha import GlobalPointerNeZha
from models.configuration_nezha import NeZhaConfig
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME


from args import get_args
from utils import FGM,compute_kl_loss
from dataprocess import load_train_data

args=get_args()
set_seed(args.seed)



def train():

    #加载数据
    if args.train_data_file_path and os.path.isfile(args.train_data_file_path):
        train_data_file_path = args.train_data_file_path
    else:
        train_data_file_path = os.path.join(args.train_data_path, "train.txt")

    datalist, label_set = load_train_data(train_data_file_path)

    #伪标签数据
    fake_datalist = []
    if args.do_fake_label:
        assert os.path.isfile(os.path.join(args.fake_train_data_path,
                                           args.fake_train_data_name)), "you must input the fake label data path"
        fake_datalist, _ = load_train_data(os.path.join(args.fake_train_data_path, args.fake_train_data_name))

    datalist.extend(fake_datalist)

    train_data_df = pd.DataFrame(datalist)
    train_data_df["label"] = train_data_df["label"].apply(lambda x: str(x)) #？label 不是整个list嘛

    #取400条作为dev data
    if len(fake_datalist) > 0:
        dev_data_df = pd.DataFrame(datalist[-(400 + len(fake_datalist)):-len(fake_datalist)])
    else:
        dev_data_df = pd.DataFrame(datalist[-400:])
    dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

    label_list = sorted(list(label_set))

    #Dataset
    ner_train_dataset = Dataset(train_data_df, categories=label_list)
    ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)

    tokenizer = Tokenizer(vocab=args.model_name, max_seq_len=128)

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    #model

    config = NeZhaConfig.from_pretrained(args.model_name,
                                         num_labels=len(ner_train_dataset.cat2id))

    torch.cuda.empty_cache()

    dl_module = GlobalPointerNeZha.from_pretrained(args.model_name,
                                                   config=config)

    optimizer = get_default_model_optimizer(dl_module, weight_decay=args.weight_decay)
    if args.swa:
        optimizer = SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq, swa_lr=args.swa_lr)

    t_total = (len(ner_train_dataset) / args.batch_size) // args.gradient_accumulation_steps * args.num_epoches
    warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    model = Task(module=dl_module,
                 optimizer=optimizer,
                 loss_function='gpce',
                 scheduler=scheduler,
                 cuda_device=args.cuda_device)

    fgm = FGM(model.module, args.adv_name, args.adv_epsilon)
    model.fit(ner_train_dataset,
              ner_dev_dataset,
              lr=args.learning_rate,
              epochs=args.num_epoches,
              batch_size=args.batch_size,
              grad_clip=args.max_grad_norm,
              fgm=fgm if args.do_adv else None,
              compute_kl_loss=compute_kl_loss if args.rdrop else None,
              rdrop_rate=args.rdrop_rate)

    return model, tokenizer, ner_train_dataset.cat2id


def save_model(model, tokenizer):
    model_state_dict_dir = os.path.join(args.model_save_dir,
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

    train_args = {
        "model_name": args.model_name,
        "num_epoches": args.num_epoches,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "warmup_proportion": args.warmup_proportion,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    if args.do_adv:
        train_args.update({
            "do_adv": True,
            "adv_epsilon": args.adv_epsilon,
            "adv_name": args.adv_name
        })
    if args.swa:
        train_args.update({
            "swa": True,
            "swa_start": args.swa_start,
            "swa_freq": args.swa_freq,
            "swa_lr": args.swa_lr
        })
    if args.do_fake_label:
        train_args.update({
            "fake_train_data_path": args.fake_train_data_path
        })
    if args.rdrop:
        train_args.update({
            "rdrop_rate": args.rdrop_rate
        })
    with open(os.path.join(model_state_dict_dir, "train_config.json"), "wt", encoding="utf-8") as f:
        f.write(json.dumps(train_args, indent=4, ensure_ascii=False))
    model_to_save = model.module if hasattr(model, 'module') else model
    # 保存模型权重pytorch_model.bin
    torch.save(model_to_save.state_dict(), os.path.join(model_state_dict_dir, WEIGHTS_NAME))
    # 保存模型配置文件config.json
    model_to_save.config.to_json_file(os.path.join(model_state_dict_dir, CONFIG_NAME))
    # 保存vocab.txt
    tokenizer.vocab.save_vocabulary(model_state_dict_dir)
    return model_state_dict_dir


if __name__ == "__main__":
    model, tokenizer, cat2id = train()
    model_state_dict_dir = save_model(model, tokenizer)



