"""
prepare_data.py - Group 7, Information Security 3(2-1), MNSUAM
Loads the selected CIC-IDS2017 files, cleans them, maps labels to three
classes (Normal, DDoS, Brute-force) and saves a processed CSV.
Run locally:  python src/prepare_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42                      # fixed seed for reproducibility
np.random.seed(SEED)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Only these three files are used. Wednesday (DoS), Thursday and the other
# Friday files are excluded on purpose (out of project scope).
# Change the file names below if your downloaded files are named differently.
FILES = {
    "Monday":      "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday":     "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Friday_DDoS": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

# Mapping from original labels to the three target classes (Table 2).
LABEL_MAP = {
    "BENIGN":      "Normal",
    "FTP-Patator": "Brute-force",
    "SSH-Patator": "Brute-force",
    "DDoS":        "DDoS",
}

# Identifier and time columns (present only in the 84-column version).
ID_TIME_COLS = ["Flow ID", "Source IP", "Source Port",
                "Destination IP", "Timestamp"]


def load_one(day, filename):
    """Read one CSV, strip header spaces, remove the duplicate header column."""
    df = pd.read_csv(RAW_DIR / filename, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()          # " Label" -> "Label"
    if "Fwd Header Length.1" in df.columns:      # duplicated column in CIC-IDS2017
        same = (df["Fwd Header Length"] == df["Fwd Header Length.1"]).all()
        print(f"[{day}] Fwd Header Length.1 identical to original: {same}")
        if same:
            df = df.drop(columns=["Fwd Header Length.1"])
    df["source_file"] = day                      # kept for analysis only
    return df


def main():
    log = []                                     # rows remaining after each step

    # 1. Load the three files and show the ORIGINAL label counts per file.
    data = pd.concat([load_one(d, f) for d, f in FILES.items()],
                     ignore_index=True)
    data["Label"] = data["Label"].astype(str).str.strip()
    raw_counts = pd.crosstab(data["Label"], data["source_file"], margins=True)
    print("\nOriginal label counts per file:\n", raw_counts)
    raw_counts.to_csv(OUT_DIR / "raw_label_counts.csv")
    log.append(("raw rows loaded", len(data)))

    # 2. Preserve the original label; stop if an unexpected label appears.
    data = data.rename(columns={"Label": "Label_original"})
    unknown = set(data["Label_original"]) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected labels, decide how to handle: {unknown}")
    data.insert(0, "row_id", np.arange(len(data)))   # key to join metadata later

    # 3. Separate identifier/time columns (if present) and drop them from the data.
    meta_cols = [c for c in ID_TIME_COLS if c in data.columns]
    if meta_cols:
        data[["row_id"] + meta_cols].to_csv(OUT_DIR / "ids_meta.csv", index=False)
    data = data.drop(columns=meta_cols)
    print("Identifier/time columns dropped:", meta_cols or "none present")

    # 4. Make every feature numeric; infinite values become NaN.
    skip = ["row_id", "Label_original", "source_file"]
    feats = [c for c in data.columns if c not in skip]
    data[feats] = data[feats].apply(pd.to_numeric, errors="coerce")
    data[feats] = data[feats].replace([np.inf, -np.inf], np.nan)
    print("Rows with missing/infinite values:", int(data[feats].isna().any(axis=1).sum()))
    print("Rows with negative Flow Duration (reported, not removed):",
          int((data["Flow Duration"] < 0).sum()))

    # 5. Remove rows with missing/infinite values, then exact duplicates.
    data = data.dropna(subset=feats)
    log.append(("after removing missing/infinite rows", len(data)))
    data = data.drop_duplicates(subset=feats + ["Label_original"])
    log.append(("after removing duplicate rows", len(data)))

    # 6. Map original labels to the three target classes.
    data["Class"] = data["Label_original"].map(LABEL_MAP)

    # 7. Class counts and cleaning log (these feed Section 2.2 and Section 4).
    counts = pd.crosstab([data["Label_original"], data["Class"]],
                         data["source_file"], margins=True)
    print("\nCounts after cleaning:\n", counts)
    print("\nClass distribution (%):\n",
          (data["Class"].value_counts(normalize=True) * 100).round(2))
    counts.to_csv(OUT_DIR / "class_counts.csv")
    pd.DataFrame(log, columns=["step", "rows"]).to_csv(
        OUT_DIR / "cleaning_log.csv", index=False)

    # 8. Save the processed dataset.
    data.to_csv(OUT_DIR / "ids_processed.csv", index=False)
    print("\nSaved:", OUT_DIR / "ids_processed.csv", data.shape)


if __name__ == "__main__":
    main()