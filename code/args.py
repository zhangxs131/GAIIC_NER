import os
import argparse


def get_args():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="ner")
    parser.add_argument("--train_data_path", default=os.path.join(BASE_DIR, "data/contest_data/train_data"),
                        type=str, help="train data file path")
    parser.add_argument("--train_data_file_path", default="",
                        type=str, help="train data file path")
    parser.add_argument("--model_name", default="data/pretrain_model/nezha-cn-base", type=str)
    parser.add_argument("--model_save_dir", default="output_model", type=str)

    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_epoches", default=6, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--cuda_device", default=4, type=int, help="the number of cuda to use")
    parser.add_argument("--seed", default=42, type=int)

    # adversarial training
    parser.add_argument("--do_adv", action="store_true",
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=0.5, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--type", default="train", type=str, help="train or predict")
    parser.add_argument("--predict_model_path", type=str, default="output_model/model_nezha_fgm_epoch_6.pth")

    # 伪标签训练
    parser.add_argument("--do_fake_label", action="store_true", help="whether to use fake label training.")
    parser.add_argument("--fake_train_data_path", type=str, default=os.path.join(BASE_DIR, "data/orther"))
    parser.add_argument("--fake_train_data_name", type=str, default="fake_train_data_20000.txt")

    # 使用swa
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa_start", default=10, type=int)
    parser.add_argument("--swa_freq", default=5, type=int)
    parser.add_argument("--swa_lr", default=0.01, type=float)

    # 使用rdrop
    parser.add_argument("--rdrop", action="store_true")
    parser.add_argument("--rdrop_rate", default=0.5, type=float)

    args = parser.parse_args()

    return args
