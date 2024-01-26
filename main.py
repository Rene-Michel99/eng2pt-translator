import json
import argparse
import numpy as np
import tensorflow as tf

from utils import *
from CustomModel import Translator
from CustomLosses import CustomLosses


MAX_VOCAB_SIZE = 5000
UNITS = 1024


def _save_vocab(path, text_processor):
    with open(path, "w") as f:
        f.write(json.dumps({"vocab": text_processor.get_vocabulary()}))


def _load_vocab(vocab_type):
    with open(f"./Data/{vocab_type}", "r") as f:
        return json.loads(f.read())["vocab"]


def _training(dataset_path):
    raw_data = []
    with open(dataset_path, 'r') as f:
        for line in f.readlines():
            raw_data.append(line.split('\t')[:2])

    eng_data = np.array([item[0] for item in raw_data])
    pt_data = np.array([item[1] for item in raw_data])

    BUFFER_SIZE = len(raw_data)
    BATCH_SIZE = 64

    is_train = np.random.uniform(size=(len(pt_data),)) < 0.8

    print("Processing data to optimized training...")

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((eng_data[is_train], pt_data[is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((eng_data[~is_train], pt_data[~is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    for example_context_strings, example_target_strings in train_raw.take(1):
        print(example_context_strings[:5])
        print()
        print(example_target_strings[:5])
        break

    input_text_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_eng,
        max_tokens=MAX_VOCAB_SIZE,
        ragged=True
    )

    output_text_processor = tf.keras.layers.TextVectorization(
        standardize=preprocess_pt,
        max_tokens=MAX_VOCAB_SIZE,
        ragged=True
    )

    input_text_processor.adapt(train_raw.map(lambda en, pt: en))
    output_text_processor.adapt(train_raw.map(lambda en, pt: pt))

    def process_text(context, target):
        context = input_text_processor(context).to_tensor()
        target = output_text_processor(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out

    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
        print(ex_context_tok[0, :10].numpy())
        print()
        print(ex_tar_in[0, :10].numpy())
        print(ex_tar_out[0, :10].numpy())

    model = Translator(
        UNITS,
        input_text_processor,
        output_text_processor,
    )
    logits = model((ex_context_tok, ex_tar_in))
    model.compile(
        optimizer='adam',
        loss=CustomLosses.masked_loss,
        metrics=[CustomLosses.masked_acc, CustomLosses.masked_loss]
    )
    vocab_size = 1.0 * input_text_processor.vocabulary_size()

    # That should roughly match the values returned by running a few steps of evaluation:
    model.evaluate(val_ds, steps=20, return_dict=True)
    history = model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
    )
    model.save_weights("logs/translator_weights.h5")
    print("Model saved in logs/translator_weights.h5")
    _save_vocab("./Data/input_vocab.json", input_text_processor)
    _save_vocab("./Data/output_vocab.json", output_text_processor)


def _inference(model_weights):
    input_text_processor = tf.keras.layers.TextVectorization(
        vocabulary=_load_vocab("input_vocab"),
        standardize=preprocess_eng,
        max_tokens=MAX_VOCAB_SIZE,
        split='whitespace',
        ragged=True
    )

    output_text_processor = tf.keras.layers.TextVectorization(
        vocabulary=_load_vocab("output_vocab"),
        standardize=preprocess_pt,
        max_tokens=MAX_VOCAB_SIZE,
        ragged=True
    )

    model = Translator(
        UNITS,
        input_text_processor,
        output_text_processor,
    )
    model.compile(
        optimizer='adam',
        loss=CustomLosses.masked_loss,
        metrics=[CustomLosses.masked_acc, CustomLosses.masked_loss]
    )
    model.load_weights(model_weights)

    input_text = str(input("Insert the text to be translated: "))

    result = model.translate([input_text])
    result = result[0].numpy().decode()
    print(result)


def main(mode, dataset_path=None, model_weights=None):
    if mode == "inference":
        print("Executing inference mode...")
        print("Loading inference model...")
        _inference(model_weights)
    elif mode == "training":
        print("Executing training mode...")
        print("Loading dataset from:", dataset_path)
        _training(dataset_path)
    else:
        print("Unkdown mode. Use 'inference' or 'training'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network to translate different languages.")
    parser.add_argument("mode", choices=["inference", "training"], help="Operation mode, inference execute single translation or training that executes the training).")
    parser.add_argument("--dataset_path", help="Path to training dataset (optional)")
    parser.add_argument("--model_weights", help="Path to trained model weights (optional)")

    args = parser.parse_args()
    main(args.mode, args.dataset_path, args.model_weights)
