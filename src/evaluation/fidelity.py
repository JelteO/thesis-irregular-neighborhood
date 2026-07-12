# fidelity.py - aggregates the occlusion drops of the audit tables into one fidelity+
# table per dataset: mean, median and % positive over the top-20 flagged entries
# ---------------------------------------------------------------

import pandas as pd


def fidelity():
    try:
        for dataset in ["schreyer", "philadelphia"]:
            audit = pd.read_csv(f"outputs/audit_table_{dataset}.csv").head(20)
            rows = []
            for col, name in [
                ("vendor_drop_pct", "gl_account / vendor"),
                ("department_drop_pct", "profit_center / department"),
            ]:
                drops = audit[col]
                rows.append(
                    {
                        "neighbour type": name,
                        "mean fid+ (%)": round(drops.mean(), 1),
                        "median fid+ (%)": round(drops.median(), 1),
                        "positive (%)": round((drops > 0).mean() * 100, 0),
                    }
                )
            table = pd.DataFrame(rows)
            table.to_csv(f"outputs/fidelity_{dataset}.csv", index=False)
    except Exception as e:
        print(f"fidelity failed: {e}")
        return False
    return True
