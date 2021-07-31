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
parser.add_argument(
    "--data", type=str, default="", help="dataset to use",
)
parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
)
parser.add_argument(
    "--steps",
    type=str,
    default=50000,
    help="Maximum steps to train the model",
)
parser.add_argument(
    "--emsize", type=int, default=128, help="size of word embeddings"
)
parser.add_argument(
    "--nhid", type=int, default=128, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
parser.add_argument(
    "--lr", type=float, default=0.001, help="initial learning rate"
)
parser.add_argument(
    "--clip", type=float, default=0.25, help="gradient clipping"
)
parser.add_argument(
    "--batch_size", type=int, default=16, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=64, help="sequence length")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument(
    "--tied",
    action="store_true",
    help="tie the word embedding and softmax weights",
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="report interval",
)
parser.add_argument(
    "--val-interval",
    type=int,
    default=100,
    metavar="N",
    help="validation interval",
)
# parser.add_argument(
#     "--save", type=str, default="model.pt", help="path to save the final model"
# )
parser.add_argument(
    "--nhead",
    type=int,
    default=2,
    help="the number of heads in the encoder/decoder of the transformer model",
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

# corpus = data.Corpus(args.data)

# # Starting from sequential data, batchify arranges the dataset into columns.
# # For instance, with the alphabet as the sequence and batch size 4, we'd get
# # ┌ a g m s ┐
# # │ b h n t │
# # │ c i o u │
# # │ d j p v │
# # │ e k q w │
# # └ f l r x ┘.
# # These columns are treated as independent by the model, which means that the
# # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# # batch processing.


# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)


eval_batch_size = 10
# train_data = batchify(corpus.train, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)
# test_data = batchify(corpus.test, eval_batch_size)


# def get_dataset(key):
#     if key == "lmd":
#         return muspy.LakhMIDIDataset(DATASET_DIR / "lmd")
#     if key == "wikifornia":
#         return muspy.WikiforniaDataset(DATASET_DIR / "wikifornia")
#     if key == "nes":
#         return muspy.NESMusicDataset(DATASET_DIR / "nes")
#     if key == "jsb":
#         return muspy.JSBChoralesDataset(DATASET_DIR / "jsb")
#     if key == "maestro":
#         return muspy.MAESTRODatasetV2(DATASET_DIR / "maestro")
#     if key == "hymnal":
#         return muspy.HymnalDataset(DATASET_DIR / "hymnal")
#     if key == "hymnal_tune":
#         return muspy.HymnalDataset(DATASET_DIR / "hymnal_tune")
#     if key == "music21":
#         return muspy.MusicDataset(DATASET_DIR / "music21")
#     if key == "music21jsb":
#         return muspy.Music21Dataset("bach")
#     if key == "nmd":
#         return muspy.NottinghamDatabase(DATASET_DIR / "nmd")
#     if key == "essen":
#         return muspy.EssenFolkSongDatabase(DATASET_DIR / "essen")
#     raise ValueError("Unrecognized dataset name.")


# data_map = {
#     "lmd": muspy.LakhMIDIDataset(DATASET_DIR / "lmd"),
#     "wikifornia": muspy.WikiforniaDataset(DATASET_DIR / "wikifornia"),
#     "nes": muspy.NESMusicDataset(DATASET_DIR / "nes"),
#     "jsb": muspy.JSBChoralesDataset(DATASET_DIR / "jsb"),
#     "maestro": muspy.MAESTRODatasetV2(DATASET_DIR / "maestro"),
#     "hymnal": muspy.HymnalDataset(DATASET_DIR / "hymnal"),
#     "hymnal_tune": muspy.HymnalDataset(DATASET_DIR / "hymnal_tune"),
#     "music21": muspy.MusicDataset(DATASET_DIR / "music21"),
#     "music21jsb": muspy.Music21Dataset("bach"),
#     "nmd": muspy.NottinghamDatabase(DATASET_DIR / "nmd"),
#     "essen": muspy.EssenFolkSongDatabase(DATASET_DIR / "essen"),
# }
# if args.data not in data_map:
#     raise ValueError("Unrecognized dataset name.")
# torch_dataset = data_map[args.data].to_pytorch_dataset(factory=factory)

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


if args.data not in DATASET_KEYS:
    raise ValueError("Unrecognized dataset name.")
music_dataset = muspy.MusicDataset(DATASET_DIR / args.data)
split_filename = DATASET_DIR / args.data / "splits.txt"
# music_dataset = get_dataset(args.data)
pytorch_datasets = music_dataset.to_pytorch_dataset(
    factory=factory, split_filename=split_filename, splits=(0.8, 0.1, 0.1)
)
train_data = DataLoader(
    pytorch_datasets["train"],
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.n_jobs,
)
val_data = DataLoader(
    pytorch_datasets["validation"],
    batch_size=eval_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.n_jobs,
)
test_data = DataLoader(
    pytorch_datasets["test"],
    batch_size=eval_batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.n_jobs,
)

###############################################################################
# Build the model
###############################################################################

# ntokens = len(corpus.dictionary)
ntokens = eos + 1
if args.model == "Transformer":
    model = model.TransformerModel(
        ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout
    ).to(device)
elif args.model == "TransformerL":
    model = model.TransformerModel(
        ntokens, args.emsize, 4, args.nhid, args.nlayers, args.dropout
    ).to(device)
else:
    model = model.RNNModel(
        args.model,
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayers,
        args.dropout,
        args.tied,
    ).to(device)

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


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


n_val_trials = 100


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    val_total_loss = 0.0
    # ntokens = len(corpus.dictionary)
    if "Transformer" not in args.model:
        val_hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        # for i in range(0, data_source.size(0) - 1, args.bptt):
        #     data, targets = get_batch(data_source, i)
        val_loader = iter(data_source)
        val_trial = 0
        while val_trial < n_val_trials:
            try:
                val_data_, val_targets = val_loader.next()
            except StopIteration:
                val_loader = iter(data_source)
                val_data_, val_targets = val_loader.next()
            # for data, targets in data_source:
            val_data_ = val_data_.t().to(device)
            val_targets = val_targets.t().reshape(-1).to(device)
            if "Transformer" in args.model:
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
    # return total_loss / (len(data_source) - 1)


train_loader = iter(train_data)


# Loop over epochs.
# lr = args.lr
best_val_loss = None
optim = torch.optim.Adam(model.parameters(), args.lr)

(MODEL_DIR / args.data).mkdir(exist_ok=True)
model_filename = str(MODEL_DIR / args.data / "{}.pt".format(args.model))

(LOG_DIR / args.data).mkdir(exist_ok=True)
(LOG_DIR / args.data / args.model).mkdir(exist_ok=True)
log_file = open(str(LOG_DIR / args.data / args.model / "train.log"), "w")
val_log_file = open(
    str(LOG_DIR / args.data / args.model / "validation.log"), "w"
)
log_file.write("# step, loss, perplexity\n")
val_log_file.write("# step, loss, perplexity\n")

step = 0


# At any point you can hit Ctrl + C to break out of training early.
try:
    print("Start training...")
    while step < args.steps:

        # Turn on training mode which enables dropout.
        total_loss = 0.0
        start_time = time.time()
        cycle_start_time = time.time()
        # ntokens = len(corpus.dictionary)
        if "Transformer" not in args.model:
            hidden = model.init_hidden(args.batch_size)
        # for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        #     data, targets = get_batch(train_data, i)
        while step < args.steps:
            try:
                data, targets = train_loader.next()
            except StopIteration:
                train_loader = iter(train_data)
                data, targets = train_loader.next()
            # for batch, (data, targets) in enumerate(train_data):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            data = data.t().to(device)
            targets = targets.t().reshape(-1).to(device)

            model.train()
            model.zero_grad()
            if "Transformer" in args.model:
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            # Logger
            if step and step % args.log_interval == 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print(
                    "| step {:5d} | ms/batch {:5.2f} | loss {:5.2f} | "
                    "ppl {:8.2f}".format(
                        step,
                        elapsed * 1000 / args.log_interval,
                        cur_loss,
                        math.exp(cur_loss),
                    )
                )
                log_file.write(
                    "{}, {}, {}\n".format(step, cur_loss, math.exp(cur_loss))
                )
                total_loss = 0.0
                start_time = time.time()

            # Validation
            if step and step % args.val_interval == 0:
                val_loss = evaluate(val_data)
                print("-" * 80)
                print(
                    "| step {:5d} | time: {:5.2f}s | valid loss {:5.2f} | "
                    "valid ppl {:8.2f}".format(
                        step,
                        (time.time() - cycle_start_time),
                        val_loss,
                        math.exp(val_loss),
                    )
                )
                val_log_file.write(
                    "{}, {}, {}\n".format(step, val_loss, math.exp(val_loss))
                )
                cycle_start_time = time.time()
                print("-" * 80)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(model_filename, "wb") as f:
                        # with open(args.save, "wb") as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                # else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                # lr /= 4.0

            step += 1

except KeyboardInterrupt:
    print("-" * 80)
    print("Exiting from training early")

log_file.close()
val_log_file.close()

# Load the best saved model.
with open(model_filename, "rb") as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"]:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print("=" * 80)
print(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}"
    "".format(test_loss, math.exp(test_loss))
)
print("=" * 80)
