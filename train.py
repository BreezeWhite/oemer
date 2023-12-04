import sys
import time
import os

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

model = sys.argv[1]

def prepare_classifier_data():
    if not os.path.exists("train_data"):
        classifier.collect_data(2000)


if model == "segnet":
    model = train.train_model("ds2_dense", data_model="segnet", learning_rate=5e-6, steps=1500)
    filename = get_model_base_name("segnet")
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
elif model == "unet":
    model = train.train_model("CvcMuscima-Distortions", data_model="unet", learning_rate=5e-6)
    filename = get_model_base_name("unet")
    os.makedirs(filename)
    write_text_to_file(model.to_json(), os.path.join(filename, "arch.json"))
    model.save_weights(os.path.join(filename, "weights.h5"))
elif model == "rests_above8":
    prepare_classifier_data()
    classifier.train_rests_above8(get_model_base_name(model))
elif model == "rests":
    prepare_classifier_data()
    classifier.train_rests(get_model_base_name(model))
elif model == "all_rests":
    prepare_classifier_data()
    classifier.train_all_rests(get_model_base_name(model))
elif model == "sfn":
    prepare_classifier_data()
    classifier.train_sfn(get_model_base_name(model))
elif model == "clef":
    prepare_classifier_data()
    classifier.train_clefs(get_model_base_name(model))
elif model == "notehead":
    prepare_classifier_data()
    classifier.train_noteheads(get_model_base_name(model))
elif model == "rhythm":
    classifier.train_rhythm(get_model_base_name(model))
else:
    print("Unknown model: " + model)
    sys.exit(1)