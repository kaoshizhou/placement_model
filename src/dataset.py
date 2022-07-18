from torch.utils.data import DataLoader, Dataset

class LayoutDataset(Dataset):
    def __init__(self, config, split) -> None:
        super().__init__()
        assert split in ["train", "valid", "test"], f"illegal data split {split}, split must be one of `train`, `valid`, `test`."

        if split == "train":
            self.source_path = config.args.train_source_file
            self.target_path = config.args.train_target_file
        elif split == "valid":
            self.source_path = config.args.valid_source_file
            self.target_path = config.args.valid_target_file
        elif split == "test":
            self.source_path = config.args.test_source_file
            self.target_path = config.args.test_target_file

        self._build()
    
    def _build(self):
        with open(self.source_path, 'r') as f_source:
            source = f_source.read()
            source = source.strip().split('\n')
        with open(self.target_path, 'r') as f_target:
            target = f_target.read()
            target = target.strip().split('\n')
        
        l_source, l_target = len(source), len(target)
        assert l_source == l_target, "lines in source and target file of dataset should have same length."

        self.source = source
        self.target = target
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        src_text = self.source[index]
        tgt_text = self.target[index]

        return {
            'source_text': src_text,
            'target_text': tgt_text,
        }
