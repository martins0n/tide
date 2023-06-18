import pathlib
import zipfile

import numpy as np
import pandas as pd
from etna.datasets import TSDataset

if __name__ == "__main__":
    file_name = pathlib.Path(__file__).parent / "electricity.csv"
    df = pd.read_csv(file_name)
    print(df.head())
    df = df.rename(columns={"date": "timestamp"})
    df = pd.melt(
        df, id_vars=["timestamp"], var_name="segment", value_name="target"
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target"] = df["target"].astype(np.float32)

    print(df.head())
    print(df.info())

    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq="1H")

    desc = tsdataset.describe()

    print(desc.head())

    assert desc[lambda x: x["num_missing"] > 0].__len__() == 0

    df.to_parquet(
        pathlib.Path(__file__).parent / "electricity_paper.parquet",
        compression="gzip",
        index=False,
    )
