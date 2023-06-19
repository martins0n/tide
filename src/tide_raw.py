from typing import Any
from tide import TiDE
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd
from etna.datasets import TSDataset
from etna.transforms import StandardScalerTransform, DateFlagsTransform


class TiDEModel(pl.LightningModule):
    def __init__(self, lr=1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        
        self.tide = TiDE(*args, **kwargs)
        self.loss = nn.MSELoss()
        self.lr = lr

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
        self.log("test_mae", nn.L1Loss()(y_hat, batch["decoder_target"]))
        return loss
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss(y_hat, batch["decoder_target"])
        self.log("test_loss", loss)
        self.log("test_mae", nn.L1Loss()(y_hat, batch["decoder_target"]))
        return loss
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Dataset:
    def __init__(self, df, lookback, horizon, n_segments,):
        self.transformed = False
        self.n_segments = n_segments
        self.n_features = df.values.shape[1] // n_segments
        self.values = df.values.astype(np.float32).reshape(df.values.shape[0], -1, self.n_features).transpose(1, 0, 2)
        self.lookback = lookback
        self.horizon = horizon
        self.len = (len(df.values) - self.lookback - self.horizon + 1) * self.n_segments


    def __getitem__(self, idx: int):
        idx_segment = idx % self.n_segments
        idx = idx // self.n_segments
        encoder_covariates = self.values[idx_segment, idx : idx + self.lookback, :-1]
        decoder_covariates = self.values[idx_segment, idx + self.lookback : idx + self.lookback + self.horizon, :-1]
        attributes = np.array([idx % self.n_segments], np.float32)
        decoder_target = self.values[
            idx_segment, idx + self.lookback : idx + self.lookback + self.horizon, -1
        ].reshape(-1, 1)
        encoder_target = self.values[
            idx_segment, idx : idx + self.lookback, -1
        ].reshape(-1, 1)
        return dict(
            decoder_covariates=decoder_covariates,
            encoder_covariates=encoder_covariates,
            attributes=attributes,
            decoder_target=decoder_target,
            encoder_target=encoder_target,
        )
    
    def __len__(self):
        return self.len

if __name__ == "__main__":
    
    
    df = pd.read_parquet("/Users/marti/Projects/tide/data/pattern.parquet")

    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq="D")
    
    n_segments = tsdataset.df.shape[1]
    
    lookback = 14
    horizon = 7
    
    train_size = int(len(tsdataset.raw_df) * 0.7)
    test_size = int(len(tsdataset.raw_df) * 0.2)
    val_size = len(tsdataset.raw_df) - train_size - test_size

    train_dataset, test_dataset = tsdataset.train_test_split(
        test_size=val_size + test_size
    )

    transform = [
        DateFlagsTransform(out_column="flags"),
        StandardScalerTransform(),
    ]
    train_dataset.fit_transform(transform)
    test_dataset.transform(transform)
    
    n_features = train_dataset.df.shape[1] // n_segments - 1
    train_dataset = train_dataset.to_pandas()
    test_dataset = test_dataset.to_pandas()
    

    tsdataset = pd.concat([train_dataset, test_dataset])

    borders = {
        "train": [0, train_size],
        "val": [train_size - lookback, train_size + val_size],
        "test": [len(tsdataset) - test_size - lookback, len(tsdataset)],
    }

    datasets = {
        "train": Dataset(
            tsdataset.iloc[borders["train"][0] : borders["train"][1]],
            lookback, horizon, n_segments
        ),
        "val": Dataset(
            tsdataset.iloc[borders["val"][0] : borders["val"][1]],
            lookback, horizon, n_segments
        ),
        "test": Dataset(
            tsdataset.iloc[borders["test"][0] : borders["test"][1]],
            lookback, horizon, n_segments
        ),
    }
    
    tide = TiDEModel(
        ne_blocks = 2,
        nd_blocks = 2,
        hidden_size = 32,
        covariates_size = n_features,
        p = 0.1,
        lookback = lookback,
        decoder_output_size = horizon * 8,
        temporal_decoder_hidden_size = 32,
        feature_projection_output_size = 32,
        feature_projection_hidden_size = 32,
        horizon = horizon,
        static_covariates_size = 1,
    )
    
    batch_size = 32
    
    train_dataloader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(datasets["val"], batch_size=batch_size)
    test_dataloader = DataLoader(datasets["test"], batch_size=batch_size)
    
    trainer = pl.Trainer(max_epochs=100)
    
    trainer.fit(tide, train_dataloader, val_dataloader)
    
    trainer.test(tide, test_dataloader)
    
    
    