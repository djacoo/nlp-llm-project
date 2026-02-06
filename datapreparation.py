import torch
from datasets import load_dataset
from collections import Counter
import numpy as np

RANDOM_SEED = 42


def load_and_prepare_data():
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
    ds = dataset["train"]

    # Filter only English tickets
    ds = ds.filter(lambda ex: ex["language"] == "en")

    # Select only 5 departments
    target_queues = [
        "Technical Support",
        "Customer Service",
        "Billing and Payments",
        "Sales and Pre-Sales",
        "General Inquiry",
    ]
    ds = ds.filter(lambda ex: ex["queue"] in target_queues)

    # Shuffle and split train/val/test
    ds = ds.shuffle(seed=RANDOM_SEED)
    train_test = ds.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_SEED)

    train_ds = train_test["train"]
    val_ds = test_valid["train"]
    test_ds = test_valid["test"]

    # Label mapping for discriminative methods
    label_list = sorted(list(set(train_ds["queue"])))
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    print("Label distribution (train):")
    print(Counter(train_ds["queue"]))

    return train_ds, val_ds, test_ds, label_list, label2id, id2label
