
from typing import Union, Any, List, Tuple, Dict

import cv2
from cv2.typing import RotatedRect
import numpy as np
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering


BBox = Tuple[int, int, int, int]

def get_bbox(data: ndarray) -> List[BBox]:
    contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = (x, y, x+w, y+h)
        bboxes.append(box)
    return bboxes


def get_center(bbox: Union[BBox, ndarray]) -> Tuple[int, int]:
    cen_y = int(round((bbox[1] + bbox[3]) / 2))
    cen_x = int(round((bbox[0] + bbox[2]) / 2))
    return cen_x, cen_y


def get_edge(data):
    if len(data.shape) == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = cv2.GaussianBlur(data, (5, 5), 0)
    data = cv2.Canny(data, 10, 80)
    return data


def merge_nearby_bbox(bboxes: List[BBox], distance: float, x_factor: int = 1, y_factor: int = 1) -> List[BBox]:
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, compute_full_tree=True)
    centers = np.array([(bb[0]+bb[2], bb[1]+bb[3]) for bb in bboxes]) / 2
    centers[:, 0] *= x_factor  # Increase/decrease the x distance
    centers[:, 1] *= y_factor  # Increase/decrease the y distance
    model.fit(centers)
    labels = np.unique(model.labels_)
    new_box = []
    for label in labels:
        idx = np.where(model.labels_ == label)[0]
        xs = [[bboxes[i][0], bboxes[i][2]] for i in idx]
        ys = [[bboxes[i][1], bboxes[i][3]] for i in idx]
        x1, x2 = np.min(xs), np.max(xs)
        y1, y2 = np.min(ys), np.max(ys)
        box = (x1, y1, x2, y2)
        new_box.append(box)
    return new_box


def rm_merge_overlap_bbox(
    bboxes: List[BBox], 
    mode: str = 'remove', 
    overlap_ratio: float = 0.5
) -> List[BBox]:
    assert mode in ['remove', 'merge'], mode

    pts = np.array([(box[2], box[3]) for box in bboxes])
    max_x, max_y = np.max(pts[:, 0]), np.max(pts[:, 1])
    mask = np.zeros((max_y, max_x), dtype=np.uint16)

    box_infos: List[Dict[str, Any]] = []
    for box in bboxes:
        area_size = (box[3]-box[1]) * (box[2]-box[0])
        box_infos.append({
            "bbox": box,
            "area_size": area_size
        })
    box_infos = sorted(box_infos, key=lambda info: info["area_size"], reverse=True)

    records = {}
    for idx, box_info in enumerate(box_infos):
        box = box_info["bbox"]
        region = mask[box[1]:box[3], box[0]:box[2]]
        vals = set(np.unique(region))
        if 0 in vals:
            vals.remove(0)

        if len(vals) == 0:
            records[idx+1] = box_info
            mask[box[1]:box[3], box[0]:box[2]] = idx + 1
            continue

        area_size = box_info["area_size"]
        valid = True
        for val in vals:
            tar_info = records[val]
            tar_size = tar_info["area_size"]
            overlap_size = region[region==val].size
            assert tar_size >= area_size, f"{tar_size}, {area_size}, {val}"
            ratio = overlap_size / area_size
            if ratio > overlap_ratio:
                if mode == "merge":
                    box = box_info['bbox']
                    tar_box = tar_info['bbox']
                    top_x = min(box[0], tar_box[0])
                    top_y = min(box[1], tar_box[1])
                    bt_x = max(box[2], tar_box[2])
                    bt_y = max(box[3], tar_box[3])
                    records[val]['bbox'] = (top_x, top_y, bt_x, bt_y)
                valid = False
                break

        if valid:
            records[idx+1] = box_info
            mask[box[1]:box[3], box[0]:box[2]] = idx + 1

    valid_box = [info['bbox'] for info in records.values()]
    return valid_box


def find_lines(data: ndarray, min_len: int = 10, max_gap: int = 20) -> List[BBox]:
    assert len(data.shape) == 2, f"{type(data)} {data.shape}"

    lines = cv2.HoughLinesP(data.astype(np.uint8), 1, np.pi/180, 50, None, min_len, max_gap)
    new_line = []
    if lines is not None:
        for line in lines:
            line = line[0]
            top_x, bt_x = (line[0], line[2]) if line[0] < line[2] else (line[2], line[0])
            top_y, bt_y = (line[1], line[3]) if line[1] < line[3] else (line[3], line[1])
            new_line.append((top_x, top_y, bt_x, bt_y))
    return new_line


def draw_lines(lines: ndarray, ori_img: ndarray, width: int = 3) -> ndarray:
    img = ori_img.copy()
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), width, cv2.LINE_AA)
    return img


def to_rgb_img(data: ndarray) -> ndarray:
    if len(data.shape) >=3:
        return data
    
    img = np.ones(data.shape + (3,), dtype=np.uint8) * 255
    idx = np.where(data > 0)
    img[idx[0], idx[1]] = 0
    return img


def draw_bounding_boxes(
    bboxes: List[BBox], 
    img: ndarray, 
    color: Tuple[int, int, int] = (0, 255, 0), 
    width: int = 2, 
    inplace: bool = False
) -> ndarray:
    if len(img.shape) < 3:
        img = to_rgb_img(img)
    if not inplace:
        img = np.array(img)
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, width)
    return img


def get_rotated_bbox(data: ndarray) -> List[RotatedRect]:
    contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        bboxes.append(cv2.minAreaRect(cnt))
    return bboxes


def draw_rotated_bounding_boxes(bboxes, img, color=(0, 255, 0), width=2, inplace=False):
    if len(img.shape) < 3:
        img = to_rgb_img(img)
    if not inplace:
        img = np.array(img)
    for rbox in bboxes:
        box = cv2.boxPoints(rbox).astype(np.int64)
        cv2.drawContours(img, [box], 0, color, width)
        # cv2.putText(img, f"{rbox[1][0]:.2f} / {rbox[1][1]:.2f} / {rbox[2]:.2f}",
        #  (box[1][0]+2, box[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    return img
    
