import pathlib
import zipfile

import numpy as np
import pandas as pd
from etna.datasets import TSDataset

# show all columns
pd.set_option("display.max_columns", None)


FREQ = {
    "ETTh1": "H",
    "ETTh2": "H",
    "ETTm1": "15T",
    "ETTm2": "15T",
    "electricity": "1H",
    "traffic": "1H",
    "weather": "10T",
}

if __name__ == "__main__":
    
    for file_name in (pathlib.Path(__file__).parent).glob("**/*.csv"):
        print(file_name)
        df = pd.read_csv(file_name)
        df = df.rename(columns={"date": "timestamp"})
        df = pd.melt(
            df, id_vars=["timestamp"], var_name="segment", value_name="target"
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["target"] = df["target"].astype(np.float32)

        print(df.head())
        print(df.info())

        # show duplicates
        print("duplicates:")
        print(df[df.duplicated(subset=["timestamp", "segment"], keep=False)])

        # drop duplicates
        df = df.drop_duplicates(subset=["timestamp", "segment"], keep="first")
        
        if file_name.stem == "weather":
            segments = df["segment"].unique()
            missed_range = pd.date_range("2020-05-29 09:40:00", "2020-05-29 11:00:00", freq="10T")
            
            missed = pd.DataFrame({
                "timestamp": list(missed_range) * len(segments),
                "segment": np.repeat(segments, len(missed_range)),
                "target": 0
            })
            df = pd.concat([df, missed])
            df = df.sort_values(["timestamp", "segment"])
        
        print(df.head())

        tsdataset = TSDataset.to_dataset(df)
        tsdataset = TSDataset(tsdataset, freq=FREQ[file_name.stem])


        desc = tsdataset.describe()
        
        
        assert desc[lambda x: x["num_missing"] > 0].__len__() == 0

        df.to_parquet(
            pathlib.Path(__file__).parent / f"{file_name.stem}.parquet",
            compression="gzip",
            index=False,
        )
