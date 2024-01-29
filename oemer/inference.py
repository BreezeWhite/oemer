import os
import pickle
from PIL import Image
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray

from oemer import MODULE_PATH


def resize_image(image: Image.Image):
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    w, h = image.size
    pis = w * h
    if 3000000 <= pis <= 435000:
        return image
    lb = 3000000 / pis
    ub = 4350000 / pis
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    print(tar_w, tar_h)
    return image.resize((tar_w, tar_h))


def inference(
    model_path: str, 
    img_path: str, 
    step_size: int = 128, 
    batch_size: int = 16, 
    manual_th: Optional[Any] = None, 
    use_tf: bool = False
) -> Tuple[ndarray, ndarray]:
    if use_tf:
        import tensorflow as tf

        arch_path = os.path.join(model_path, "arch.json")
        w_path = os.path.join(model_path, "weights.h5")
        model = tf.keras.models.model_from_json(open(arch_path, "r").read())
        model.load_weights(w_path)
        input_shape = model.input_shape
        output_shape = model.output_shape
    else:
        import onnxruntime as rt

        onnx_path = os.path.join(model_path, "model.onnx")
        metadata = pickle.load(open(os.path.join(model_path, "metadata.pkl"), "rb"))
        providers = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = rt.InferenceSession(onnx_path, providers=providers)
        output_names = metadata['output_names']
        input_shape = metadata['input_shape']
        output_shape = metadata['output_shape']

    # Collect data
    image_pil = Image.open(img_path)
    if "GIF" != image_pil.format:
        # Tricky workaround to avoid random mistery transpose when loading with 'Image'.
        image_cv = cv2.imread(img_path)
        image_pil = Image.fromarray(image_cv)

    image_pil = image_pil.convert("RGB")
    image = np.array(resize_image(image_pil))
    win_size = input_shape[1]
    data = []
    for y in range(0, image.shape[0], step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            hop = image[y:y+win_size, x:x+win_size]
            data.append(hop)

    # Predict
    pred = []
    for idx in range(0, len(data), batch_size):
        print(f"{idx+1}/{len(data)} (step: {batch_size})", end="\r")
        batch = np.array(data[idx:idx+batch_size])
        out = model.predict(batch) if use_tf else sess.run(output_names, {'input': batch})[0]
        pred.append(out)

    # Merge prediction patches
    output_shape = image.shape[:2] + (output_shape[-1],)
    out = np.zeros(output_shape, dtype=np.float32)
    mask = np.zeros(output_shape, dtype=np.float32)
    hop_idx = 0
    for y in range(0, image.shape[0], step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            batch_idx = hop_idx // batch_size
            remainder = hop_idx % batch_size
            hop = pred[batch_idx][remainder]
            out[y:y+win_size, x:x+win_size] += hop
            mask[y:y+win_size, x:x+win_size] += 1
            hop_idx += 1

    out /= mask
    if manual_th is None:
        class_map = np.argmax(out, axis=-1)
    else:
        assert len(manual_th) == output_shape[-1]-1, f"{manual_th}, {output_shape[-1]}"
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(out[..., idx+1]>th, 1, 0)

    return class_map, out


def predict(region: ndarray, model_name: str) -> str:
    if np.max(region) == 1:
        region *= 255
    m_info = pickle.load(open(os.path.join(MODULE_PATH, f"sklearn_models/{model_name}.model"), "rb"))
    model = m_info['model']
    w = m_info['w']
    h = m_info['h']
    region = np.array(Image.fromarray(region.astype(np.uint8)).resize((w, h)))
    pred = model.predict(region.reshape(1, -1))
    return m_info['class_map'][pred[0]]


if __name__ == "__main__":
    img_path = "/home/kohara/omr/test_imgs/wind2.jpg"
    model_path = "./checkpoints/seg_net"
    class_map, out = inference(model_path, img_path)
