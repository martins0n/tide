import pathlib
import random
import warnings
from dataclasses import dataclass

import hydra
import hydra_slayer
import numpy as np
import pandas as pd
import torch
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

from src.tide_etna import TiDEModel

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

    df = pd.read_parquet(FILE_FOLDER / "data" / cfg.dataset.name)

    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq=cfg.dataset.freq)

    train_dataset, test_dataset = tsdataset.train_test_split(
        test_size=cfg.experiment.horizon * cfg.experiment.n_folds
    )

    transform = [StandardScalerTransform()]
    train_dataset.fit_transform(transform)
    test_dataset.transform(transform)
    train_dataset = train_dataset.to_pandas()
    test_dataset = test_dataset.to_pandas()

    tsdataset = pd.concat([train_dataset, test_dataset])

    tsdataset = TSDataset(tsdataset, freq=cfg.dataset.freq, known_future="all")

    tslogger.add(
        WandbLogger(project="tide", config=OmegaConf.to_container(cfg, resolve=True))
    )

    if "model" in cfg:
        horizon = cfg.model.horizon
        lookback = cfg.model.lookback
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

        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer_params = {
            "max_epochs": max_epochs,
            "accelerator": cfg.accelerator,
            "callbacks": [lr_monitor],
        }

        pipeline = Pipeline(
            model=TiDEModel(
                encoder_length=lookback,
                decoder_length=horizon,
                lr=lr,
                ne_blocks=ne_blocks,
                nd_blocks=nd_blocks,
                hidden_size=hidden_size,
                covariates_size=covariates_size,
                p=dropout_level,
                lookback=lookback,
                temporal_decoder_hidden_size=temporal_decoder_hidden_size,
                decoder_output_size=decoder_output_size,
                feature_projection_output_size=feature_projection_output_size,
                feature_projection_hidden_size=feature_projection_hidden_size,
                horizon=horizon,
                trainer_params=trainer_params,
                static_covariates_size=static_covariates_size,
                train_batch_size=cfg.model.train_batch_size,
                test_batch_size=cfg.model.test_batch_size,
                split_params={
                    "train_size": cfg.model.train_size,
                },
            ),
            transforms=[
                TimeFlagsTransform(
                    minute_in_hour_number=True, hour_number=True, out_column="time"
                ),
                DateFlagsTransform(
                    day_number_in_week=True,
                    day_number_in_month=True,
                    is_weekend=False,
                    out_column="date",
                ),
                StandardScalerTransform(
                    in_column=[
                        "time_minute_in_hour_number",
                        "time_hour_number",
                        "date_day_number_in_week",
                        "date_day_number_in_month",
                    ]
                ),
            ],
            horizon=horizon,
        )

    elif "baseline" in cfg:
        pipeline = hydra_slayer.get_from_params(
            **OmegaConf.to_container(cfg.baseline, resolve=True)
        )
    else:
        raise ValueError("No model or baseline specified")

    metrics_df, forecast, _ = pipeline.backtest(
        tsdataset, metrics=[MSE(), MAE(), SMAPE()], n_folds=cfg.experiment.n_folds
    )

    print(metrics_df.head())
    print(metrics_df.mean())


if __name__ == "__main__":
    run_pipeline()
