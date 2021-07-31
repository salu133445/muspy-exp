# coding: utf-8
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
import model

sys.path.append("/home/herman/git/muspy/")
import muspy

LOG_DIR = Path("/home/herman/workspace/muspy_exp/word_language_model/logs/")
MODEL_DIR = Path(
    "/home/herman/workspace/muspy_exp/word_language_model/models/"
)
DATASET_DIR = Path("/data4/herman/muspy-new/downsampled/")
DATASET_KEYS = [
    "lmd",
    "wikifonia",
    "nes",
    "jsb",
    "maestro",
    "hymnal",
    "hymnal_tune",
    "music21",
    "music21jsb",
    "nmd",
    "essen",
]


parser = argparse.ArgumentParser(description="MusPy Experiment")
# parser.add_argument(
#     "-d", "--data", type=str, default="", help="dataset to use",
# )
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="LSTM",
    help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--bptt", type=int, default=64, help="sequence length")
parser.add_argument(
    "--batch_size", type=int, default=10, metavar="N", help="batch size"
)
parser.add_argument(
    "--trials", type=int, default=100, metavar="N", help="number of trials"
)
parser.add_argument(
    "--n_jobs", type=int, default=1, help="number of workers for data loader"
)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

eval_batch_size = args.batch_size

n_timestep = args.bptt
eos = 356


def factory(music):
    # Get event representation and remove velocity events
    encoded = music.to_event_representation()
    encoded = encoded[encoded < eos].astype(np.int)

    # Pad to meet the desired length
    if len(encoded) > n_timestep:
        start = np.random.randint(encoded.shape[0] - n_timestep + 1)
        encoded = encoded[start : (start + n_timestep)]
        encoded = np.append(encoded, eos)
    elif len(encoded) < n_timestep:
        to_concat = np.ones(n_timestep - encoded.shape[0] + 1, np.int)
        to_concat.fill(eos)
        encoded = np.concatenate((encoded, to_concat))
    else:
        encoded = np.append(encoded, 129)

    return encoded[:-1], encoded[1:]


###############################################################################
# Build the model
###############################################################################

ntokens = eos + 1

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


n_val_trials = args.trials


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    val_total_loss = 0.0

    if args.model != "Transformer":
        val_hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        val_loader = iter(data_source)
        val_trial = 0
        while val_trial < n_val_trials:
            try:
                val_data_, val_targets = val_loader.next()
            except StopIteration:
                val_loader = iter(data_source)
                val_data_, val_targets = val_loader.next()

            val_data_ = val_data_.t().to(device)
            val_targets = val_targets.t().reshape(-1).to(device)

            if args.model == "Transformer":
                val_output = model(val_data_)
                val_output = val_output.view(-1, ntokens)
            else:
                val_output, val_hidden = model(val_data_, val_hidden)
                val_hidden = repackage_hidden(val_hidden)

            val_total_loss += (
                len(val_data_) * criterion(val_output, val_targets).item()
            )
            val_trial += 1

    return val_total_loss / (n_val_trials * eval_batch_size - 1)


(LOG_DIR / "test").mkdir(exist_ok=True)
(LOG_DIR / "test" / args.model).mkdir(exist_ok=True)
log_file = open(
    str(LOG_DIR / "test" / args.model / "cross-dataset-test.log"), "w"
)
log_file.write("# src, target, loss, perplexity\n")

print("=" * 80)

for src_dataset in DATASET_KEYS:

    print("-" * 80)
    print("Using model trained on dataset {}".format(src_dataset))
    model_filename = str(MODEL_DIR / src_dataset / "{}.pt".format(args.model))

    for tgt_dataset in DATASET_KEYS:

        # Load the best saved model.
        with open(model_filename, "rb") as f:
            model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            # Currently, only rnn model supports flatten_parameters function.
            if args.model in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
                model.rnn.flatten_parameters()

        music_dataset = muspy.MusicDataset(DATASET_DIR / tgt_dataset)
        split_filename = DATASET_DIR / tgt_dataset / "splits.txt"
        pytorch_datasets = music_dataset.to_pytorch_dataset(
            factory=factory,
            split_filename=split_filename,
            splits=(0.8, 0.1, 0.1),
        )
        test_data = DataLoader(
            pytorch_datasets["test"],
            batch_size=eval_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.n_jobs,
        )

        # Run on test data.
        test_loss = evaluate(test_data)

        print(
            "| Dataset {:20} | test loss {:5.2f} | test ppl {:8.2f}"
            "".format(tgt_dataset, test_loss, math.exp(test_loss))
        )
        log_file.write(
            "{}, {}, {}, {}\n".format(
                src_dataset, tgt_dataset, test_loss, math.exp(test_loss)
            )
        )

    print("-" * 80)

print("=" * 80)
log_file.close()
