# split.py - single source of truth for the split train/val/test
# in:label tensor (533009,) with 0=regular 1=local 2=global
# out:index tensors, train 426327 / val 53340 / test 53342
# ---------------------------------------------------------------

import torch


def make_entry_split(y_entry, seed=42, train_frac=0.8, val_frac=0.1):
    # regulars 80/10/10
    # anomalies never in train (clean training), half to val, half to test
    gen = torch.Generator().manual_seed(seed)

    # why: anomalies never go into train. A reconstruction model should only
    # learn normal behavior, and with just 100 anomalies in 533k entries a
    # proportional split would leave ~10 for val and ~10 for test, which is
    # too few to select or measure on. Now both sets get 50 (15 local, 35 global)

    # regular
    regular_ids = (y_entry == 0).nonzero(as_tuple=True)[0]
    shuffle = torch.randperm(len(regular_ids), generator=gen)
    regular_ids = regular_ids[shuffle]
    n_regular = len(regular_ids)
    n_train = int(train_frac * n_regular)
    n_val = int(val_frac * n_regular)
    train_ids = regular_ids[:n_train]
    val_regular = regular_ids[n_train : n_train + n_val]
    test_regular = regular_ids[n_train + n_val :]

    # local anomalies (label 1)
    local_ids = (y_entry == 1).nonzero(as_tuple=True)[0]
    shuffle = torch.randperm(len(local_ids), generator=gen)
    local_ids = local_ids[shuffle]
    half_local = len(local_ids) // 2
    val_local = local_ids[:half_local]
    test_local = local_ids[half_local:]

    # global anomalies (label 2)
    global_ids = (y_entry == 2).nonzero(as_tuple=True)[0]
    shuffle = torch.randperm(len(global_ids), generator=gen)
    global_ids = global_ids[shuffle]
    half_global = len(global_ids) // 2
    val_global = global_ids[:half_global]
    test_global = global_ids[half_global:]

    val_ids = torch.cat([val_regular, val_local, val_global])
    test_ids = torch.cat([test_regular, test_local, test_global])

    n_val_anomalies = len(val_local) + len(val_global)
    n_test_anomalies = len(test_local) + len(test_global)

    print(f"split: `train {len(train_ids)}")
    print(f"split: val {len(val_ids)} ({n_val_anomalies} anomalies)")
    print(f"split: test {len(test_ids)} ({n_test_anomalies} anomalies)")

    return train_ids, val_ids, test_ids


# why: DOMINANT builds a dense NxN adjacency and cannot score all 53342 test entries,
# so the model comparison runs on a fixed subset:
# all 50 test anomalies + 10000 sampled test regulars = 10050 entries,
# same order for every model
#
# data:returned tensor (10050,), anomalies first, then sampled regulars
def sample_common_test(y_entry, test_ids, n_regular=10_000, seed=42):
    test_labels = y_entry[test_ids]

    anomaly_ids = test_ids[test_labels != 0]
    regular_ids = test_ids[test_labels == 0]

    gen = torch.Generator().manual_seed(seed)
    shuffle = torch.randperm(len(regular_ids), generator=gen)
    sampled_regulars = regular_ids[shuffle[:n_regular]]

    common_ids = torch.cat([anomaly_ids, sampled_regulars])
    return common_ids
