import random
import os
from PIL import Image, ImageColor
from multiprocessing import Process, Queue

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import augly.image as imaugs

from .build_label import build_label, close_lines
from .models.unet import semantic_segmentation, u_net
from .constant_min import CHANNEL_NUM



def get_cvc_data_paths(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")
    
    dirs = ["curvature", "ideal", "interrupted", "kanungo", "rotated", "staffline-thickness-variation-v1",
        "staffline-thickness-variation-v2", "staffline-y-variation-v1", "staffline-y-variation-v2",
        "thickness-ratio", "typeset-emulation", "whitespeckles"]

    data = []
    for dd in dirs:
        dir_path = os.path.join(dataset_path, dd)
        folders = os.listdir(dir_path)
        for folder in folders:
            data_path = os.path.join(dir_path, folder)
            imgs = os.listdir(os.path.join(data_path, "image"))
            for img in imgs:
                img_path = os.path.join(data_path, "image", img)
                staffline = os.path.join(data_path, "gt", img)
                symbol_path = os.path.join(data_path, "symbol", img)
                data.append([img_path, staffline, symbol_path])

    return data


def get_deep_score_data_paths(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found, download the dataset first.")
    
    imgs = os.listdir(os.path.join(dataset_path, "images"))
    paths = []
    for img in imgs:
        image_path = os.path.join(dataset_path, "images", img)
        seg_path = os.path.join(dataset_path, "segmentation", img.replace(".png", "_seg.png"))
        paths.append((image_path, seg_path))
    return paths


def preprocess_image(img_path):
    image = Image.open(img_path).convert("1")
    params = {}

    if image.mode == "1":
        # The input image contains only one channel.
        arr = np.array(image)
        out = np.zeros(arr.shape + (3,), dtype=np.uint8)
        bg_is_white = np.count_nonzero(arr) > (arr.size * 0.7)
        bg_idx = np.where(arr==bg_is_white)

        # Change background color
        hue = random.randint(19, 60)
        sat = random.randint(0, 15)
        val = random.randint(70, 100)
        color = ImageColor.getrgb(f"hsv({hue}, {sat}%, {val}%)")
        out[bg_idx[0], bg_idx[1]] = color
        image = Image.fromarray(out)
        params['bg_color'] = {'hue': hue, 'saturation': sat, 'value': val}

    aug_image = image.copy()

    #random_black_spots = random.randint(0, 5)
    #for _ in range(random_black_spots):
    #    spot_image = random.randint(1, 5)
    #    aug_image = imaugs.overlay_emoji(aug_image, emoji_path="black_spot" + str(spot_image) + ".png", 
    #                                     emoji_size=random.uniform(0.01, 0.05), 
    #                                     x_pos=random.uniform(0.01, 0.99), 
    #                                     y_pos=random.uniform(0.01, 0.99))

    # Color jitter
    bright = (7 + random.randint(0, 6)) / 10  # 0.7~1.3
    saturation = (5 + random.randint(0, 7)) / 10  # 0.5~1.2
    contrast = (5 + random.randint(0, 10)) / 10  # 0.5~1.5
    aug_image = imaugs.color_jitter(
        aug_image, brightness_factor=bright, saturation_factor=saturation, contrast_factor=contrast)
    params['color_jitter'] = {'brightness': bright, 'saturation': saturation, 'contrast': contrast}

    # Blur
    rad = random.choice(np.arange(0.0001, 2.1, 0.5))
    aug_image = imaugs.blur(aug_image, radius=rad)
    params['blur_radius'] = rad

    # Pixel shuffle, kind of adding noise
    factor = random.choice(np.arange(0.0001, 0.26, 0.05))
    aug_image = imaugs.shuffle_pixels(aug_image, factor=factor)
    params['pixel_shuffle_factor'] = factor

    # Image quality
    qa = random.randint(0, 100)
    aug_image = imaugs.encoding_quality(aug_image, quality=qa)
    params['image_quality'] = qa

    # Opacity
    # level = random.randint(6, 10) / 10
    # aug_image = imaugs.opacity(aug_image, level=level)
    # params['opacity'] = level

    # Pixelize (pretty similar to blur?)
    rat = random.randint(3, 10) / 10
    aug_image = imaugs.pixelization(aug_image, ratio=rat)
    params['pixelize_ratio'] = rat

    # Add noise
    # var = random.randint(0, 5) / 100
    # aug_image = imaugs.random_noise(aug_image, var=var)
    # params['noise_variance'] = var

    return aug_image, params


def batch_transform(img, trans_func):
    if isinstance(img, Image.Image):
        return trans_func(img)

    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3

    ch_num = img.shape[2]
    result = []
    for i in range(ch_num):
        tmp_img = Image.fromarray(img[..., i].astype(np.uint8))
        tmp_img = trans_func(tmp_img)
        result.append(np.array(tmp_img))
    return np.dstack(result)

    
class MultiprocessingDataLoader:
    def __init__(self, num_worker: int):
        self._queue: Queue = Queue(maxsize=20)
        self._dist_queue: Queue = Queue(maxsize=30)
        self._process_pool = []
        for _ in range(num_worker):
            processor = Process(target=self._preprocess_image)
            processor.daemon = True
            self._process_pool.append(processor)
        self._pdist = Process(target=self._distribute_process)
        self._pdist.daemon = True

    def _start_processes(self):
        if not self._pdist.is_alive():
            self._pdist.start()
        for process in self._process_pool:
            if not process.is_alive():
                process.start()

    def _terminate_processes(self):
        self._pdist.terminate()
        for process in self._process_pool:
            process.terminate()


    def _distribute_process(self):
        pass

    def _preprocess_image(self):
        pass


class DataLoader(MultiprocessingDataLoader):
    def __init__(self, feature_files, win_size=256, num_samples=100, min_step_size=0.2, num_worker=4):
        super().__init__(num_worker)
        self.feature_files = feature_files
        random.shuffle(self.feature_files)
        self.win_size = win_size
        self.num_samples = num_samples

        if isinstance(min_step_size, float):
            min_step_size = max(min(abs(min_step_size), 1), 0.01)
            self.min_step_size = round(win_size * min_step_size)
        else:
            self.min_step_size = max(min(abs(min_step_size), win_size), 2)

        self.file_idx = 0

    def _distribute_process(self):
        while True:
            paths = self.feature_files[self.file_idx]
            self._dist_queue.put(paths)
            self.file_idx += 1
            if self.file_idx == len(self.feature_files):
                random.shuffle(self.feature_files)
                self.file_idx = 0

    def _preprocess_image(self):
        while True:
            if not self._queue.full():
                inp_img_path, staff_img_path, symbol_img_path = self._dist_queue.get()

                # Preprocess image with transformations that won't change view.
                image, _ = preprocess_image(inp_img_path)

                # Random resize
                ratio = random.choice(np.arange(0.2, 1.21, 0.1))
                tar_w = int(ratio * image.size[0])
                tar_h = int(ratio * image.size[1])
                image = imaugs.resize(image, width=tar_w, height=tar_h)
                staff_img_array = cv2.imread(staff_img_path)
                staff_img_array = cv2.cvtColor(staff_img_array, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                staff_img_array = close_lines(staff_img_array)
                staff_img = Image.fromarray(staff_img_array)
                staff_img = imaugs.resize(staff_img, width=tar_w, height=tar_h)
                symbol_img = imaugs.resize(symbol_img_path, width=tar_w, height=tar_h)

                # Random perspective transform
                seed = random.randint(0, 1000)
                np.float = float  # Monkey patch to workaround removal of np.float
                perspect_trans = lambda img: imaugs.perspective_transform(img, seed=seed, sigma=70)
                image = np.array(perspect_trans(image))  # RGB image
                staff_img = np.array(perspect_trans(staff_img))  # 1-bit mask
                symbol_img = np.array(perspect_trans(symbol_img))  # 1-bit mask
                staff_img = np.where(staff_img, 1, 0)
                symbol_img = np.where(symbol_img, 1, 0)

                self._queue.put([image, staff_img, symbol_img, ratio])

    def __iter__(self):
        samples = 0

        self._start_processes()

        while samples < self.num_samples:
            image, staff_img, symbol_img, ratio = self._queue.get()

            start_x, start_y = 0, 0
            max_y = image.shape[0] - self.win_size
            max_x = image.shape[1] - self.win_size
            while (start_x < max_x) and (start_y < max_y):
                y_range = range(start_y, start_y+self.win_size)
                x_range = range(start_x, start_x+self.win_size)
                index = np.ix_(y_range, x_range)
                # Can't use two 'range' inside the numpy array for indexing. Details refer to the following:
                # https://stackoverflow.com/questions/30020143/indexing-slicing-a-2d-numpy-array-using-the-range-arange-function-as-the-argumen
                feat = image[index]
                staff = staff_img[index]
                symbol = symbol_img[index]
                neg = np.ones_like(staff) - staff - symbol
                label = np.stack([neg, staff, symbol], axis=-1)

                yield feat, label

                y_step = random.randint(round(self.min_step_size*ratio), round(self.win_size*ratio))
                x_step = random.randint(round(self.min_step_size*ratio), round(self.win_size*ratio))
                start_y = min(start_y + y_step, max_y)
                start_x = min(start_x + x_step, max_x)

        self._terminate_processes()

    def get_dataset(self, batch_size, output_types=None, output_shapes=None):
        def gen_wrapper():
            for data in self:
                yield data

        if output_types is None:
            output_types = (tf.uint8, tf.float32)

        if output_shapes is None:
            output_shapes = ((self.win_size, self.win_size, 3), (self.win_size, self.win_size, 3))

        return tf.data.Dataset.from_generator(
                gen_wrapper, output_types=output_types, output_shapes=output_shapes
            ) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)


class DsDataLoader(MultiprocessingDataLoader):
    def __init__(self, feature_files, win_size=256, num_samples=100, step_size=0.5, num_worker=4):
        super().__init__(num_worker)
        self.feature_files = feature_files
        random.shuffle(self.feature_files)
        self.win_size = win_size
        self.num_samples = num_samples

        if isinstance(step_size, float):
            step_size = max(abs(step_size), 0.01)
            self.step_size = round(win_size * step_size)
        else:
            self.step_size = max(abs(step_size), 2)

        self.file_idx = 0

    def _distribute_process(self):
        while True:
            paths = self.feature_files[self.file_idx]
            self._dist_queue.put(paths)
            self.file_idx += 1
            if self.file_idx == len(self.feature_files):
                random.shuffle(self.feature_files)
                self.file_idx = 0

    def _preprocess_image(self):
        while True:
            if not self._queue.full():
                inp_img_path, seg_img_path = self._dist_queue.get()

                # Preprocess image with transformations that won't change view.
                image, _ = preprocess_image(inp_img_path)
                label = build_label(seg_img_path)

                # Random resize
                ratio = random.choice(np.arange(0.2, 1.21, 0.1))
                tar_w = int(ratio * image.size[0])
                tar_h = int(ratio * image.size[1])
                trans_func = lambda img: imaugs.resize(img, width=tar_w, height=tar_h)
                image = batch_transform(image, trans_func)
                label = batch_transform(label, trans_func)

                # Random perspective transform
                seed = random.randint(0, 1000)
                np.float = float  # Monkey patch to workaround removal of np.float
                perspect_trans = lambda img: imaugs.perspective_transform(img, seed=seed, sigma=70)
                image = np.array(batch_transform(image, perspect_trans))  # RGB image
                label = np.array(batch_transform(label, perspect_trans))

                self._queue.put([image, label, ratio])

    def __iter__(self):
        samples = 0

        self._start_processes()

        while samples < self.num_samples:
            image, label, ratio = self._queue.get()

            # Discard bottom spaces that has no contents.
            staff = label[..., 1]
            yidx, _ = np.where(staff>0)
            if len(yidx) > 0:
                max_y = min(np.max(yidx) + 100, image.shape[0])
            else:
                max_y = image.shape[0]

            max_y = max_y - self.win_size
            max_x = image.shape[1] - self.win_size
            grid_x = range(0, max_x, round(self.step_size*ratio))
            grid_y = range(0, max_y, round(self.step_size*ratio))
            meshgrid = np.meshgrid(grid_x, grid_y, indexing='ij')
            coords = np.dstack(meshgrid).reshape(-1, 2)
            random.shuffle(coords)
            for start_x, start_y in coords:
                y_range = range(start_y, start_y+self.win_size)
                x_range = range(start_x, start_x+self.win_size)
                index = np.ix_(y_range, x_range)

                # Can't use two 'range' inside the numpy array for indexing. Details refer to the following:
                # https://stackoverflow.com/questions/30020143/indexing-slicing-a-2d-numpy-array-using-the-range-arange-function-as-the-argumen
                feat = image[index]
                ll = label[index]
                yield feat, ll

        self._terminate_processes()
    def get_dataset(self, batch_size, output_types=None, output_shapes=None):
        def gen_wrapper():
            for data in self:
                yield data

        if output_types is None:
            output_types = (tf.uint8, tf.float32)

        if output_shapes is None:
            output_shapes = ((self.win_size, self.win_size, 3), (self.win_size, self.win_size, CHANNEL_NUM))

        return tf.data.Dataset.from_generator(
                gen_wrapper, output_types=output_types, output_shapes=output_shapes
            ) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)


def lr_scheduler(epoch, lr, update_after=5, dec_every=3, dec_rate=0.5):
    if epoch >= update_after and (epoch - update_after) % dec_every == 0:
        lr *= dec_rate
    return max(lr, 5e-8)


class WarmUpLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr=0.1, warm_up_steps=1000, decay_step=3000, decay_rate=0.25, min_lr=1e-8):
        self.init_lr = init_lr
        self.warm_up_steps = warm_up_steps
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.min_lr = min_lr

        self.warm_step_size = (init_lr - min_lr) / warm_up_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm_lr = self.min_lr + self.warm_step_size * step

        offset = step - self.warm_up_steps
        cycle = offset // self.decay_step
        start_lr = self.init_lr * tf.pow(self.decay_rate, cycle)
        end_lr = start_lr * self.decay_rate
        step_size = (start_lr - end_lr) / self.decay_step
        lr = start_lr - (offset - cycle * self.decay_step) * step_size
        true_lr = tf.where(offset > 0, lr, warm_lr)
        return tf.maximum(true_lr, self.min_lr)

    def get_config(self):
        return {
            "warm_up_steps": self.warm_up_steps,
            "decay_step": self.decay_step,
            "decay_rate": self.decay_rate,
            "min_lr": self.min_lr
        }


def focal_tversky_loss(y_true, y_pred, fw=0.7, alpha=0.7, smooth=1., gamma=0.75):
    tp_weight = 0.4  # Reduce the influence of true positive samples (mostly background).
    tp = tf.reduce_sum(y_true * y_pred) * tp_weight
    fn = tf.reduce_sum(y_true * (1-y_pred))
    fp = tf.reduce_sum((1-y_true) * y_pred)
    tversky = 1 - (tp + smooth) / (tp + alpha*fn + (1-alpha)*fp + smooth)
    t_loss = tf.pow(tversky, gamma)
    focal_loss = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred)
    return fw*focal_loss + (1-fw)*t_loss


def train_model(
    dataset_path,
    win_size=288,
    train_val_split=0.1,
    learning_rate=5e-4,
    epochs=15,
    steps=1000,
    batch_size=8,
    val_steps=200,
    val_batch_size=8,
    early_stop=8,
    data_model="segnet"
):
    if data_model == "segnet":
        feat_files = get_deep_score_data_paths(dataset_path)
    else:
        feat_files = get_cvc_data_paths(dataset_path)
    random.shuffle(feat_files)
    split_idx = round(train_val_split * len(feat_files))
    train_files = feat_files[split_idx:]
    val_files = feat_files[:split_idx]

    print(f"Loading dataset. Train/validation: {len(train_files)}/{len(val_files)}")
    if data_model == "segnet":
        train_data = DsDataLoader(
                train_files,
                win_size=win_size,
                num_samples=epochs*steps*batch_size
            ) \
            .get_dataset(batch_size)
        val_data = DsDataLoader(
                val_files,
                win_size=win_size,
                num_samples=epochs*val_steps*val_batch_size
            ) \
            .get_dataset(val_batch_size)
        model = semantic_segmentation(win_size=win_size, out_class=CHANNEL_NUM)
    else:
        train_data = DataLoader(
                train_files,
                win_size=win_size,
                num_samples=epochs*steps*batch_size
            ) \
            .get_dataset(batch_size)
        val_data = DataLoader(
                val_files,
                win_size=win_size,
                num_samples=epochs*val_steps*val_batch_size
            ) \
            .get_dataset(val_batch_size)
        model = u_net(win_size=win_size, out_class=3)

    print("Initializing model")
    optim = tf.keras.optimizers.Adam(learning_rate=WarmUpLearningRate(learning_rate))
    #loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    #loss = tf.keras.losses.CategoricalCrossentropy()
    loss = tfa.losses.SigmoidFocalCrossEntropy()
    model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=early_stop, monitor='val_accuracy'),
        tf.keras.callbacks.ModelCheckpoint("seg_unet", save_weights_only=False, monitor='val_accuracy')
    ]

    print("Start training")
    try:
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            callbacks=callbacks
        )
        return model
    except Exception as e:
        print(e)
        return model


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


def inference(model_path, img_path, step_size=128, batch_size=16, manual_th=None):
    arch_path = os.path.join(model_path, "arch.json")
    w_path = os.path.join(model_path, "weights.h5")
    model = tf.keras.models.model_from_json(open(arch_path, "r").read())
    model.load_weights(w_path)
    input_shape = model.input_shape[1:]

    # Collect data
    # Tricky workaround to avoid random mistery transpose when loading with 'Image'.
    image = cv2.imread(img_path)
    image = Image.fromarray(image).convert("RGB")
    image = np.array(resize_image(image))
    win_size = input_shape[0]
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
        out = model.predict(batch)
        pred.append(out)

    # Merge prediction patches
    output_shape = image.shape[:2] + (model.output_shape[-1],)
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
        assert len(manual_th) == model.output_shape[-1]-1, f"{manual_th}, {model.output_shape[-1]}"
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(out[..., idx+1]>th, 1, 0)

    return class_map, out


def draw_bbox(pred):
    img = np.zeros(pred.shape + (3,), dtype=np.uint8)
    idx = np.where(pred < 1)
    img[idx[0], idx[1]] = (255, 255, 255)
    contours, hierachy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img


def morph_notehead(note, size=(11, 8)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    out = cv2.erode(note.astype(np.uint8), kernel)
    return cv2.dilate(out, kernel)


if __name__ == "__main__":
    dataset_path = "/mnt/data/dataset/CvcMuscima-Distortions"
    dataset_path = "/media/kohara/ADATA HV620S/dataset/ds2_dense"
    # dataset_path = "/media/ds2_dense"

    #manual_th = [0.5, 0.3, 0.3]
    manual_th = None

    f_name = "../test_imgs/River/2.jpg"
    class_map, out = inference("staffline_only", f_name, manual_th=manual_th)
