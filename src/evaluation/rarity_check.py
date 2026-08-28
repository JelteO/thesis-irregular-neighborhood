# rarity_check.py - ground truth check on the feature attribution
# in: outputs/audit_table_schreyer.csv and the processed raw data
# out: outputs/rarity_check_schreyer.csv plus a printed hit-rate
# ---------------------------------------------------------------
# For each of the top 20 entries, the field that the explanation layer
# blames is looked up in the raw data.
#
# A global anomaly carries a value that exists owhere else,
# so a count of exactly one means the layer pointed at the injected value and
# not at an arbitrary field.


import pandas as pd


def rarity_check():
    """Look up how often the blamed value occurs in the full dataset"""
    try:
        raw_data = pd.read_csv("data/processed/fraud_dataset_processed.csv")
        audit_table = pd.read_csv("outputs/audit_table_schreyer.csv")

        rarity_rows = []
        for _, audit_entry in audit_table.head(20).iterrows():
            feature_name = audit_entry["top_feature"]

            # Skip numerical features because their values are not meaningful categories
            if feature_name.startswith("numerical"):
                continue

            # Get the value of the audited feature for this specific entry
            entry_value = raw_data.loc[audit_entry["entry_id"], feature_name]

            # Count how many times this value appears in the entire dataset
            value_frequency = int((raw_data[feature_name] == entry_value).sum())

            rarity_rows.append(
                {
                    "rank": audit_entry["rank"],
                    "entry_id": audit_entry["entry_id"],
                    "top_feature": feature_name,
                    "value": entry_value,
                    "value_count": value_frequency,
                    "ground_truth_hit": value_frequency == 1,
                }
            )

        rarity_table = pd.DataFrame(rarity_rows)
        rarity_table.to_csv("outputs/rarity_check_schreyer.csv", index=False)

        hit_count = int(rarity_table["ground_truth_hit"].sum())
        print(f"rarity check hit-rate: {hit_count}/{len(rarity_table)}")
    except Exception as error:
        print(f"rarity check failed: {error}")
        return False
    return True
