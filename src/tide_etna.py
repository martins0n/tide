from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from etna.models.base import DeepBaseModel, DeepBaseNet
from pandas import DataFrame

from src.tide import TiDE, TiDEBatch


class TiDENet(DeepBaseNet):
    def __init__(
        self,
        lr: float = 1e-2,
        loss: nn.Module = nn.MSELoss(),
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()

        self.tide = TiDE(**kwargs)
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params or {}
        self.scheduler_params = scheduler_params or {}

    def forward(self, x):
        return self.tide(x)

    def step(self, batch: TiDEBatch, *args, **kwargs):
        y_hat = self(batch)

        decoder_target = batch["decoder_target"]

        loss = self.loss(y_hat, decoder_target)
        return loss, decoder_target, y_hat

    def make_samples(
        self, df: DataFrame, encoder_length: int, decoder_length: int
    ) -> Iterable[dict]:
        max_sequence_length = encoder_length + decoder_length
        number_of_sequences = int(len(df) // max_sequence_length)
        if len(df) == max_sequence_length:
            sequence_start_idx = np.array([0])
        else:
            sequence_start_idx = np.random.randint(
                0, len(df) - max_sequence_length, size=number_of_sequences * 2
            )

        samples = []

        for idx in sequence_start_idx:
            sample = dict()
            view = df.iloc[idx : idx + max_sequence_length].select_dtypes(
                include=[np.number]
            )
            sample["encoder_target"] = view[["target"]][:encoder_length].values.astype(
                np.float32
            )
            sample["decoder_target"] = view[["target"]][encoder_length:].values.astype(
                np.float32
            )
            sample["encoder_covariates"] = view.drop(columns=["target"])[
                :encoder_length
            ].values.astype(np.float32)
            sample["decoder_covariates"] = view.drop(columns=["target"])[
                encoder_length:
            ].values.astype(np.float32)
            sample["attributes"] = np.array([0.0]).astype(
                np.float32
            )  # TODO: add attributes
            sample["segment"] = df["segment"].values[0]

            samples.append(sample)

        return samples

    def configure_optimizers(self):
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, **self.optimizer_params
        )

        if self.scheduler_params:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, **self.scheduler_params
            )
            return [optimizer], [scheduler]
        else:
            return optimizer


class TiDEModel(DeepBaseModel):
    def __init__(
        self,
        decoder_length: int,
        encoder_length: int,
        lr: float = 1e-3,
        loss: Optional["torch.nn.Module"] = nn.MSELoss(),
        train_batch_size: int = 16,
        test_batch_size: int = 16,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params
        super().__init__(
            net=TiDENet(
                lr=lr,
                loss=loss,
                optimizer_params=optimizer_params,
                **kwargs,
            ),
            decoder_length=decoder_length,
            encoder_length=encoder_length,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )
