from typing import Any
from tide import TiDE
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd
from etna.datasets import TSDataset
from etna.transforms import StandardScalerTransform

class TiDEModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        self.tide = TiDE(*args, **kwargs)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.tide(x)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss(y_hat, batch["decoder_target"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss(y_hat, batch["decoder_target"])
        self.log("val_loss", loss)
        return loss
    

class Dataset:
    def __init__(self, df, lookback, horizon):
        self.transformed = False
        self.values = df.values
        self.lookback = lookback
        self.horizon = horizon

    def __getitem__(self, idx):
        context = self.values[idx : idx + self.lookback]
        target = self.values[idx + self.lookback : idx + self.lookback + self.horizon]
        return {"context": context, "target": target}
    
    def __len__(self):
        return len(self.values) - self.lookback - self.horizon + 1

if __name__ == "__main__":
    
    
    df = pd.read_parquet("/Users/marti/Projects/tide/data/electricity_paper.parquet")

    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq="H")
    
    lookback = 720
    horizon = 96
    
    train_size = int(len(tsdataset.raw_df) * 0.7)
    test_size = int(len(tsdataset.raw_df) * 0.2)
    val_size = len(tsdataset.raw_df) - train_size - test_size

    train_dataset, test_dataset = tsdataset.train_test_split(
        test_size=val_size + test_size
    )

    transform = [StandardScalerTransform()]
    train_dataset.fit_transform(transform)
    test_dataset.transform(transform)
    train_dataset = train_dataset.to_pandas()
    test_dataset = test_dataset.to_pandas()

    tsdataset = pd.concat([train_dataset, test_dataset])

    
    borders = {
        "train": [0, train_size],
        "val": [train_size - lookback, train_size + val_size],
        "test": [len(tsdataset) - test_size - lookback, len(tsdataset)],
    }

    datasets = {
        "train": Dataset(tsdataset.iloc[borders["train"][0] : borders["train"][1]], lookback, horizon),
        "val": Dataset(tsdataset.iloc[borders["val"][0] : borders["val"][1]], lookback, horizon),
        "test": Dataset(tsdataset.iloc[borders["test"][0] : borders["test"][1]], lookback, horizon),
    }
    
    # el, hor = datasets["train"][datasets["train"].__len__() - 1]
    
    # assert el.shape[0] == lookback
    # assert hor.shape[0] == horizon
    
    # print(hor[-1, : 5])
    # print(tsdataset.iloc[borders["train"][0] : borders["train"][1]].iloc[-1, : 5])

    def collate_fn(batch):
        context = [item["context"] for item in batch]
        target = [item["target"] for item in batch]
        context = np.stack(context, axis=0)
        target = np.stack(target, axis=0)
        context = context.transpose(0, 2, 1)
        target = target.transpose(0, 2, 1)
        context = context.reshape(-1, context.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        context = torch.from_numpy(context)
        target = torch.from_numpy(target)
        return {"context": context, "target": target}
    
    batch_size = 8
    my_dataloader = DataLoader(datasets["test"], batch_size=batch_size, collate_fn=collate_fn)
    print(next(iter(my_dataloader))["context"].shape)