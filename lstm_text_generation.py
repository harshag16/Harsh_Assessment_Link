"""
lstm_text_generation.py
========================

Generative AI with LSTM — Word-level text generator trained on a public-domain
corpus (e.g. Shakespeare's works from Project Gutenberg).

Pipeline
--------
1. Load & clean text (lowercase, strip punctuation).
2. Tokenize into words, build a vocabulary, and turn the corpus into
   overlapping fixed-length input sequences with a "next word" target.
3. Build an Embedding -> stacked LSTM -> Dense(softmax) model in
   TensorFlow/Keras.
4. Train with a train/validation split, EarlyStopping, and ModelCheckpoint.
5. Generate new text from a seed phrase using temperature-controlled sampling.

Usage
-----
    # Train a model on a text file and generate samples
    python lstm_text_generation.py --data shakespeare.txt --epochs 40

    # Only generate from an already-trained model
    python lstm_text_generation.py --data shakespeare.txt --generate_only \
        --seed "to be or not to"

Requirements
------------
    pip install tensorflow numpy

Author: (your name)
"""

import argparse
import json
import os
import re
import string
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# TensorFlow / Keras imports are wrapped so the module can still be imported
# (e.g. for unit-testing the preprocessing functions) in environments where
# TensorFlow isn't installed yet.
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    TF_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 1. DATA PREPROCESSING
# --------------------------------------------------------------------------- #

def load_text(path: str) -> str:
    """Read a UTF-8 text file from disk."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(raw_text: str) -> str:
    """
    Lowercase the text and strip punctuation/extra whitespace, while keeping
    word boundaries intact. This is a light-touch cleanup appropriate for
    word-level language modelling (we keep apostrophes inside contractions
    like "don't" so the vocabulary doesn't fragment them into "don" + "t").
    """
    text = raw_text.lower()

    # Normalize newlines/whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove punctuation EXCEPT apostrophes used inside words (don't, it's)
    punctuation_to_strip = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punctuation_to_strip))

    # Drop stray apostrophes that aren't part of a contraction (e.g. quote marks)
    text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)

    return text.strip()


@dataclass
class SequenceData:
    """Container for the tokenized dataset and everything needed to map
    between words and integer ids."""
    tokenizer: "Tokenizer"
    vocab_size: int
    seq_length: int
    X: np.ndarray
    y: np.ndarray
    index_word: dict = field(default_factory=dict)


def build_sequences(text: str, seq_length: int = 10, step: int = 1,
                     max_vocab: int = 10000) -> SequenceData:
    """
    Tokenize `text` into words and build (input_sequence -> next_word) pairs.

    Parameters
    ----------
    text        : cleaned text (output of clean_text)
    seq_length  : number of words fed to the model to predict the next word
    step        : stride between the start of consecutive training windows
                  (step=1 gives the most training data; increase it to
                  shrink the dataset for faster experimentation)
    max_vocab   : cap on vocabulary size (keeps the softmax layer tractable)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for build_sequences(). "
                           "Install it with `pip install tensorflow`.")

    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])

    # word_index is 1-based; word_index size may exceed max_vocab, but
    # texts_to_sequences will map anything above max_vocab to <OOV>.
    encoded = tokenizer.texts_to_sequences([text])[0]
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab + 1)

    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_length, step):
        inputs.append(encoded[i: i + seq_length])
        targets.append(encoded[i + seq_length])

    X = np.array(inputs)
    y = np.array(targets)

    index_word = {idx: w for w, idx in tokenizer.word_index.items()}

    print(f"[preprocess] corpus tokens: {len(encoded):,} | "
          f"vocab size: {vocab_size:,} | training sequences: {len(X):,}")

    return SequenceData(tokenizer=tokenizer, vocab_size=vocab_size,
                         seq_length=seq_length, X=X, y=y, index_word=index_word)


def train_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.1,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split the (X, y) arrays into train/validation sets."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_val = max(1, int(len(X) * val_fraction))
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    return X_train, y_train, X_val, y_val


# --------------------------------------------------------------------------- #
# 2. MODEL DESIGN
# --------------------------------------------------------------------------- #

def build_model(vocab_size: int, seq_length: int, embedding_dim: int = 128,
                 lstm_units: Tuple[int, ...] = (256, 128),
                 dropout: float = 0.2) -> "Sequential":
    """
    Build an Embedding -> stacked LSTM -> Dense(softmax) model.

    Parameters
    ----------
    vocab_size    : size of the vocabulary (softmax output dimension)
    seq_length    : length of the input word sequence
    embedding_dim : dimensionality of the word embedding vectors
    lstm_units    : tuple of hidden sizes, one per LSTM layer. Passing more
                    than one value stacks LSTM layers (see the "bonus"
                    experiments in the README for how depth affects output).
    dropout       : dropout applied after the embedding and between LSTM
                    layers, to reduce overfitting on a modest corpus.
    """
    model = Sequential(name="lstm_text_generator")
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                         input_length=seq_length, name="embedding"))
    model.add(Dropout(dropout))

    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1  # all but the last layer
        model.add(LSTM(units, return_sequences=return_sequences,
                        name=f"lstm_{i + 1}"))
        model.add(Dropout(dropout))

    model.add(Dense(vocab_size, activation="softmax", name="output"))

    model.compile(
        loss="sparse_categorical_crossentropy",  # y is integer class ids
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# 3. TRAINING
# --------------------------------------------------------------------------- #

def train_model(model: "Sequential", X_train, y_train, X_val, y_val,
                 epochs: int = 40, batch_size: int = 128,
                 checkpoint_path: str = "best_model.keras",
                 patience: int = 4):
    """
    Train with EarlyStopping (restores best weights) and ModelCheckpoint
    (persists the best validation-loss model to disk).
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss",
                         save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# --------------------------------------------------------------------------- #
# 4. TEXT GENERATION
# --------------------------------------------------------------------------- #

def sample_with_temperature(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample an index from a probability distribution, reshaped by
    `temperature`. temperature < 1 -> more conservative/repetitive text;
    temperature > 1 -> more random/creative (and more error-prone) text.
    """
    probs = np.asarray(probs).astype("float64")
    if temperature <= 0:
        return int(np.argmax(probs))

    logits = np.log(probs + 1e-9) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    scaled_probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(scaled_probs), p=scaled_probs))


def generate_text(model, tokenizer: "Tokenizer", index_word: dict,
                   seed_text: str, seq_length: int, num_words: int = 50,
                   temperature: float = 0.8) -> str:
    """
    Iteratively predict the next word `num_words` times, each time feeding
    the growing sequence (truncated/padded to seq_length) back into the model.
    """
    result_words = seed_text.split()
    generated = clean_text(seed_text)

    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([generated])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, padding="pre")

        probs = model.predict(encoded, verbose=0)[0]
        next_id = sample_with_temperature(probs, temperature)
        next_word = index_word.get(next_id, "")

        if not next_word:
            break

        result_words.append(next_word)
        generated = generated + " " + next_word

    return " ".join(result_words)


# --------------------------------------------------------------------------- #
# 5. PERSISTENCE HELPERS (save/load tokenizer alongside the model)
# --------------------------------------------------------------------------- #

def save_artifacts(model, tokenizer: "Tokenizer", seq_length: int,
                    out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        f.write(tokenizer.to_json())
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"seq_length": seq_length}, f)
    print(f"[save] model + tokenizer written to '{out_dir}/'")


def load_artifacts(out_dir: str = "artifacts"):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    model = load_model(os.path.join(out_dir, "model.keras"))
    with open(os.path.join(out_dir, "tokenizer.json")) as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(os.path.join(out_dir, "config.json")) as f:
        seq_length = json.load(f)["seq_length"]
    index_word = {idx: w for w, idx in tokenizer.word_index.items()}
    return model, tokenizer, index_word, seq_length


# --------------------------------------------------------------------------- #
# MAIN / CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="LSTM word-level text generator")
    parser.add_argument("--data", type=str, default="sample_shakespeare.txt",
                         help="Path to a .txt corpus")
    parser.add_argument("--seq_length", type=int, default=10,
                         help="Number of words in each input sequence")
    parser.add_argument("--step", type=int, default=1,
                         help="Stride between training windows")
    parser.add_argument("--max_vocab", type=int, default=8000)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--lstm_units", type=int, nargs="+", default=[256, 128],
                         help="e.g. --lstm_units 256 128 for a 2-layer stack")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--generate_only", action="store_true",
                         help="Skip training; load a previously saved model")
    parser.add_argument("--seed", type=str, default=None,
                         help="Seed text for generation")
    parser.add_argument("--num_words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if not TF_AVAILABLE:
        raise SystemExit("TensorFlow is not installed. Run "
                          "`pip install tensorflow` and try again.")

    if args.generate_only:
        model, tokenizer, index_word, seq_length = load_artifacts(args.artifacts_dir)
    else:
        raw = load_text(args.data)
        cleaned = clean_text(raw)
        data = build_sequences(cleaned, seq_length=args.seq_length,
                                step=args.step, max_vocab=args.max_vocab)

        X_train, y_train, X_val, y_val = train_val_split(
            data.X, data.y, val_fraction=args.val_fraction)

        model = build_model(data.vocab_size, data.seq_length,
                             embedding_dim=args.embedding_dim,
                             lstm_units=tuple(args.lstm_units))
        model.summary()

        train_model(model, X_train, y_train, X_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size,
                    checkpoint_path=os.path.join(args.artifacts_dir, "best_model.keras"))

        save_artifacts(model, data.tokenizer, data.seq_length, args.artifacts_dir)
        tokenizer, index_word, seq_length = data.tokenizer, data.index_word, data.seq_length

    # --- Generate sample outputs from a few seeds -------------------------
    seeds = [args.seed] if args.seed else [
        "to be or not to",
        "shall i compare thee to",
        "friends romans countrymen lend",
    ]
    for seed in seeds:
        text = generate_text(model, tokenizer, index_word, seed,
                              seq_length=seq_length, num_words=args.num_words,
                              temperature=args.temperature)
        print("\nSEED:", seed)
        print("GENERATED:", text)


if __name__ == "__main__":
    main()
