from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from pytorch_lightning import LightningDataModule, LightningModule


class RandomDictDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        a = self.data[index]
        b = a + 2
        return {'a': a, 'b': b}

    def __len__(self):
        return self.len


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class RandomIterableDataset(IterableDataset):

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(self.count):
            yield torch.randn(self.size)


class RandomIterableDatasetWithLen(IterableDataset):

    def __init__(self, size: int, count: int):
        self.count = count
        self.size = size

    def __iter__(self):
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self):
        return self.count


class BoringDataModule(LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None
        self.checkpoint_state: Optional[str] = None

    def prepare_data(self):
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.random_train = Subset(self.random_full, indices=range(64))
            self.dims = self.random_train[0].shape

        if stage in ("fit", "validate") or stage is None:
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))

        if stage == "test" or stage is None:
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))
            self.dims = getattr(self, "dims", self.random_test[0].shape)

        if stage == "predict" or stage is None:
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))
            self.dims = getattr(self, "dims", self.random_predict[0].shape)

    def train_dataloader(self):
        return DataLoader(self.random_train)

    def val_dataloader(self):
        return DataLoader(self.random_val)

    def test_dataloader(self):
        return DataLoader(self.random_test)

    def predict_dataloader(self):
        return DataLoader(self.random_predict)

