import sys
import time
import os

import tensorflow as tf

from oemer import train
from oemer import classifier


def write_text_to_file(text, path):
    with open(path, "w") as f:
        f.write(text)

if len(sys.argv) != 2:
    print("Usage: python train.py <model_name>")
    sys.exit(1)

def get_model_base_name(model_name: str) -> str:
    timestamp = str(round(time.time()))
    return f"{model_name}_{timestamp}"

model_type = sys.argv[1]

def prepare_classifier_data():
    if not os.path.exists("train_data"):
        classifier.collect_data(2000)

if model_type == "segnet":
    model = train.train_model("ds2_dense", data_model=model_type, steps=1500, epochs=15)
    filename = get_model_base_name(model_type)
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
elif model_type == "unet":
    model = train.train_model("CvcMuscima-Distortions", data_model=model_type, steps=1500, epochs=15)
    filename = get_model_base_name(model_type)
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
elif model_type == "unet_from_checkpoint" or model_type == "segnet_from_checkpoint":
    model = tf.keras.models.load_model("seg_unet", custom_objects={"WarmUpLearningRate": train.WarmUpLearningRate})
    filename = get_model_base_name(model_type.split("_")[0])
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
elif model_type == "rests_above8":
    prepare_classifier_data()
    classifier.train_rests_above8(get_model_base_name(model_type))
elif model_type == "rests":
    prepare_classifier_data()
    classifier.train_rests(get_model_base_name(model_type))
elif model_type == "all_rests":
    prepare_classifier_data()
    classifier.train_all_rests(get_model_base_name(model_type))
elif model_type == "sfn":
    prepare_classifier_data()
    classifier.train_sfn(get_model_base_name(model_type))
elif model_type == "clef":
    prepare_classifier_data()
    classifier.train_clefs(get_model_base_name(model_type))
else:
    print("Unknown model: " + model_type)
    sys.exit(1)