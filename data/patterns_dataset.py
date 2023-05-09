import pathlib
import random

import numpy as np
from etna.datasets.datasets_generation import generate_from_patterns_df

if __name__ == "__main__":
    # set seed
    seed = 11
    np.random.seed(seed)
    random.seed(seed)

    df = generate_from_patterns_df(
        start_time="2021-01-01",
        patterns=[[1, 100, 100, 233, 1], [222, 333, 333, 333, 333, 222]],
        periods=1000,
    )

    df.to_parquet(
        pathlib.Path(__file__).parent / "pattern.parquet",
        index=False,
        compression="gzip",
    )
