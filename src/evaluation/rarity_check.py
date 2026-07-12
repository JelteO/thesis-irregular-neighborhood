import pandas as pd

raw = pd.read_csv("data/processed/fraud_dataset_processed.csv")
audit = pd.read_csv("outputs/audit_table_schreyer.csv")

# to do: wrap in function and add to pipeline in main.py (i think)
for _, entry in audit.head(20).iterrows():
    field = entry["top_feature"]

    # amounts (the numerical colums) have no single value to count, only categorical fields
    if field.startswith("numerical"):
        continue

    # lookup the actual value of the 'blamed' field for this entry
    value = raw.loc[entry["entry_id"], field]

    # count how often that value occurs in the whole dataset
    count = (raw[field] == value).sum()
    share = count / len(raw) * 100

    print(f"rank {entry['rank']:>2} | {field}={value} | {count}x ({share:.4f}%)")
