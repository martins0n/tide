import pathlib
import random
import warnings
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pytorch_lightning as pl
import hydra
import hydra_slayer
import numpy as np
import pandas as pd
import torch
import wandb
from etna.datasets import TSDataset
from etna.loggers import WandbLogger, tslogger
from etna.metrics import MAE, MSE, SMAPE
from etna.pipeline import Pipeline
from etna.transforms import (
    DateFlagsTransform,
    StandardScalerTransform,
    TimeFlagsTransform,
)
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger as plWandbLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.tide_raw import TiDEModel, Dataset
from torch.utils.data import DataLoader
from collections import defaultdict

OmegaConf.register_new_resolver("mul", lambda x, y: x * y)


FILE_FOLDER = pathlib.Path(__file__).parent.absolute()

# filter future warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class ModelConfig:
    horizon: int
    lookback: int
    ne_blocks: int
    nd_blocks: int
    hidden_size: int
    dropout_level: float
    covariates_size: int
    temporal_decoder_hidden_size: int
    decoder_output_size: int
    static_covariates_size: int
    lr: float
    max_epochs: int
    feature_projection_output_size: int
    feature_projection_hidden_size: int
    train_batch_size: int
    test_batch_size: int
    train_size: float
    layer_norm: bool


@dataclass
class DatasetConfig:
    name: str
    freq: str


@dataclass
class ExperimentConfig:
    horizon: int
    n_folds: int = 1


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    experiment: ExperimentConfig
    baseline: dict
    seed: int = 11
    accelerator: str = "cpu"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="configs", config_name="config")
def run_pipeline(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # set seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    horizon = cfg.model.horizon
    lookback = cfg.model.lookback

    df = pd.read_parquet(FILE_FOLDER / "data" / cfg.dataset.name)

    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq=cfg.dataset.freq)
    n_segments = tsdataset.df.shape[1]
    
    
    train_size = int(len(tsdataset.raw_df) * 0.7)
    test_size = int(len(tsdataset.raw_df) * 0.2)
    val_size = len(tsdataset.raw_df) - train_size - test_size

    train_dataset, test_dataset = tsdataset.train_test_split(
        test_size=val_size + test_size
    )

    transform = [
            TimeFlagsTransform(
                minute_in_hour_number=True, hour_number=True, out_column="atime"
            ),
            DateFlagsTransform(
                day_number_in_week=True,
                day_number_in_month=True,
                is_weekend=False,
                out_column="adate",
            ),
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


    tsdataset = pd.concat([train_dataset, test_dataset])


    # tslogger.add(
    #     WandbLogger(project="tide", config=OmegaConf.to_container(cfg, resolve=True))
    # )


    ne_blocks = cfg.model.ne_blocks
    nd_blocks = cfg.model.nd_blocks
    hidden_size = cfg.model.hidden_size
    dropout_level = cfg.model.dropout_level
    covariates_size = cfg.model.covariates_size
    temporal_decoder_hidden_size = cfg.model.temporal_decoder_hidden_size
    decoder_output_size = cfg.model.decoder_output_size
    static_covariates_size = cfg.model.static_covariates_size
    lr = cfg.model.lr
    max_epochs = cfg.model.max_epochs
    feature_projection_output_size = cfg.model.feature_projection_output_size
    feature_projection_hidden_size = cfg.model.feature_projection_hidden_size
    layer_norm = cfg.model.layer_norm
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer_params = {
        "max_epochs": max_epochs,
        "accelerator": cfg.accelerator,
        "callbacks": [lr_monitor],
        "logger": plWandbLogger(project="tide", config=OmegaConf.to_container(cfg, resolve=True)),
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
        lr=lr,
        ne_blocks = ne_blocks,
        nd_blocks = nd_blocks,
        hidden_size = hidden_size,
        covariates_size = covariates_size,
        p = dropout_level,
        lookback = lookback,
        decoder_output_size = decoder_output_size,
        temporal_decoder_hidden_size = temporal_decoder_hidden_size,
        feature_projection_output_size = feature_projection_output_size,
        feature_projection_hidden_size = feature_projection_hidden_size,
        horizon = horizon,
        static_covariates_size = static_covariates_size,
        layer_norm=layer_norm
    )
    
    batch_size = cfg.model.train_batch_size
    test_batch_size = cfg.model.test_batch_size
    
    train_dataloader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(datasets["val"], batch_size=test_batch_size)
    test_dataloader = DataLoader(datasets["test"], batch_size=test_batch_size)
    
    trainer = pl.Trainer(**trainer_params)
    
    trainer.fit(tide, train_dataloader, val_dataloader)
    
    trainer.test(tide, test_dataloader)
    
    pred = trainer.predict(tide, test_dataloader)
    
    results = defaultdict(list)
    
    for batch_pred, batch_target in zip(pred, test_dataloader):
        
        
        
        results["pred"].append(batch_pred)
        results["target"].append(batch_target["decoder_target"])
        results["attributes"].append(batch_target["attributes"].repeat((1, horizon)))

    
    results["pred"] = torch.cat(results["pred"], dim=0).detach().cpu().numpy()
    results["target"] = torch.cat(results["target"], dim=0).detach().cpu().numpy()
    results["attributes"] = torch.cat(results["attributes"], dim=0).detach().cpu().numpy()
    
    df = pd.DataFrame(
        {
            "pred": results["pred"].flatten(),
            "target": results["target"].flatten(),
            "attributes": results["attributes"].flatten(),
        }
    )
    
    df["time"] = df.groupby("attributes").transform('cumcount')

    mse_mean = mean_squared_error(df["target"], df["pred"])
    mae_mean = mean_absolute_error(df["target"], df["pred"])
    results = wandb.Artifact(
        "results", 
        type="dataset"
    ) 
    
    
    wandb.log({
        "MAE_mean": mae_mean,
        "MSE_mean": mse_mean
    })
    
    with TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/results.csv.gz", "wb") as f:
            df.to_csv(f, index=False, compression="gzip")
            f.flush()
        results.add_file(tmpdir + "/results.csv.gz")
    
        wandb.log_artifact(results)


if __name__ == "__main__":
    run_pipeline()
