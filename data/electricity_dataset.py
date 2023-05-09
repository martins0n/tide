import pathlib
import zipfile

import numpy as np
import pandas as pd
from etna.datasets import TSDataset

if __name__ == "__main__":
    with zipfile.ZipFile(pathlib.Path(__file__).parent / "LD2011_2014.txt.zip") as z:
        with z.open("LD2011_2014.txt") as f:
            df = pd.read_csv(f, sep=";", decimal=",")
            df = df.rename(columns={"Unnamed: 0": "timestamp"})
            df = pd.melt(
                df, id_vars=["timestamp"], var_name="segment", value_name="target"
            )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target"] = df["target"].astype(np.float32)

    print(df.head())
    print(df.info())

    tsdataset = TSDataset.to_dataset(df)
    tsdataset = TSDataset(tsdataset, freq="15min")

    desc = tsdataset.describe()

    print(desc.head())

    assert desc[lambda x: x["num_missing"] > 0].__len__() == 0

    df.to_parquet(
        pathlib.Path(__file__).parent / "electricity.parquet",
        compression="gzip",
        index=False,
    )
