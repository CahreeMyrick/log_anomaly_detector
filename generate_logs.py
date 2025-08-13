import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

def gen_ts():
    # get current time
    return datetime.now()    

def gen_level():
    # gen random number betweeen 0 and 20
    return np.random.randint(20)

def gen_user():
    # random 4 digit number for user id
    my_str = ""
    for i in range(4):
        my_str += str(np.random.randint(9))

    return my_str

def gen_action():
    acts = ['push', 'pull','commit', 'add']
    
    rand_idx = np.random.randint(3)

    return acts[rand_idx]


def gen_log(attrs: dict[str, any]):

    log = []
    for attr in attrs.keys():
        log.append(attrs[attr]())

    return log 

def gen_log_dataset(attrs: dict[str, any], num_logs=100):
    data = pd.DataFrame(columns=list(attrs.keys()))
    for i in range(num_logs):
        log = gen_log(attrs)
        data.loc[len(data)] = log

    return data

def gen_anomalies(
    data: pd.DataFrame,
    attrs: Dict[str, Any],
    frac: float = 0.05,
    seed: Optional[int] = None,
    per_row: Tuple[int, int] = (1, 2),   # corrupt 1–2 fields per anomalous row
) -> pd.DataFrame:
    """
    Injects anomalies into ~frac of rows.
    Adds columns:
      - label: 1 if row was corrupted
      - anomaly_types: comma-separated descriptions of what was corrupted
    """

    rng = np.random.default_rng(seed)
    df = data.copy()
    n = len(df)
    if n == 0:
        df["label"] = []
        df["anomaly_types"] = []
        return df

    n_rows_to_corrupt = max(1, int(np.ceil(frac * n)))
    df["label"] = 0
    anomaly_types = [[] for _ in range(n)]
    cols = list(attrs.keys())

    def corrupt_cell(row_idx: int, col: str):
        # NOTE: these are example anomalies tailored to your schema
        if col == "level":
            # out-of-range, negative, or wrong type
            choice = rng.choice(["out_of_range", "negative", "string"])
            if choice == "out_of_range":
                df.at[row_idx, col] = int(rng.integers(50, 101))
            elif choice == "negative":
                df.at[row_idx, col] = int(-rng.integers(1, 10))
            else:  # wrong type
                df.at[row_idx, col] = pd.NA
            anomaly_types[row_idx].append(f"level:{choice}")

        elif col == "user":
            # unrealistic id or non-numeric id
            choice = rng.choice(["huge", "non_numeric"])
            if choice == "huge":
                df.at[row_idx, col] = str(rng.integers(100000, 1000000))
            else:
                df.at[row_idx, col] = rng.choice(["guest", "unknown", "root"])
            anomaly_types[row_idx].append(f"user:{choice}")

        elif col == "action":
            # oov action, typo, or payload-y string
            choice = rng.choice(["oov", "typo", "payload"])
            if choice == "oov":
                df.at[row_idx, col] = rng.choice(["drop_db", "hack", "format_disk", "sudo"])
            elif choice == "typo":
                df.at[row_idx, col] = rng.choice(["cmomit", "pulll", "psuh", "addd"])
            else:
                df.at[row_idx, col] = "<script>alert(1)</script>"
            anomaly_types[row_idx].append(f"action:{choice}")

        elif col == "ts":
            # far future or far past timestamp
            choice = rng.choice(["future", "past"])
            now = pd.Timestamp(datetime.now())
            if choice == "future":
                df.at[row_idx, col] = now + pd.Timedelta(days=int(rng.integers(365, 3650)))
            else:
                df.at[row_idx, col] = now - pd.Timedelta(days=int(rng.integers(3650, 36500)))
            anomaly_types[row_idx].append(f"ts:{choice}")

        else:
            # generic: set NaN (schema anomaly)
            df.at[row_idx, col] = np.nan
            anomaly_types[row_idx].append(f"{col}:nan")

    rows = rng.choice(n, size=n_rows_to_corrupt, replace=False)
    for r in rows:
        k = int(rng.integers(per_row[0], per_row[1] + 1))
        to_corrupt = rng.choice(cols, size=min(k, len(cols)), replace=False)
        for c in to_corrupt:
            corrupt_cell(int(r), str(c))
        df.at[int(r), "label"] = 1

    df["anomaly_types"] = [", ".join(x) if x else "" for x in anomaly_types]
    return df        
        


def main():

    func_map = {
        'ts': gen_ts,
        'level': gen_level,
        'user': gen_user,
        'action': gen_action
    }

    dataset = gen_log_dataset(attrs=func_map)
    dataset = gen_anomalies(dataset, func_map, frac=0.2, seed=0)
    print(dataset.head())
    print("\nAnomaly rate:", dataset['label'].mean())
    # print(dataset.head())


if __name__ == "__main__":
    main()





