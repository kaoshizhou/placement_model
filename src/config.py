import argparse


class Config:
    def __init__(self):
        self.args = self.build_config()

    def build_config(self):
        parser = argparse.ArgumentParser()

        # ----------training args------------
        parser.add_argument("--epochs", default=50, type=int)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--weight_decay", default=0, type=float)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--warmup_ratio", default=0.1, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--distribute", default=True, type=bool, help="whether or not to use distributed training")
        parser.add_argument("--local_rank", default=-1, type=int, help="rank")


        # -----------model args---------------
        parser.add_argument("--model_name_or_path", default="t5-base", type=str, help="the backbone for training and inference")
        parser.add_argument("--output_dir", default="save_model", type=str)
        parser.add_argument("--tokenizer_path", default="t5-base", type=str, help="tokenizer directory")
        parser.add_argument("--config", type=str, default="new_config", help="config for model to train from scratch")

        # -----------data args----------------
        parser.add_argument("--shuffle", default=True, type=bool, help="shuffle or not when training")
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--train_source_file", default="./2m_data_new/train_source.txt", type=str)
        parser.add_argument("--train_target_file", default="./2m_data_new/train_target.txt", type=str)
        parser.add_argument("--valid_source_file", default="./2m_data_new/valid_source.txt", type=str)
        parser.add_argument("--valid_target_file", default="./2m_data_new/valid_target.txt", type=str)
        parser.add_argument("--test_source_file", default="./2m_data_new/test_source.txt", type=str)
        parser.add_argument("--test_target_file", default="./2m_data_new/test_target.txt", type=str)

        # parser.add_argument("--train_source_file", default="./cleaned_data/train.source", type=str)
        # parser.add_argument("--train_target_file", default="./cleaned_data/train.target", type=str)
        # parser.add_argument("--valid_source_file", default="./cleaned_data/valid.source", type=str)
        # parser.add_argument("--valid_target_file", default="./cleaned_data/valid.target", type=str)
        # parser.add_argument("--test_source_file", default="./2m_data/test_source.txt", type=str)
        # parser.add_argument("--test_target_file", default="./2m_data/test_target.txt", type=str)

        args = parser.parse_args()
        return args