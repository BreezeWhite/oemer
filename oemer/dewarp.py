import os
import pickle
import typing
from typing import List, Tuple, Any, Union
from typing_extensions import Self

import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.interpolate import interp1d, griddata
from sklearn.linear_model import LinearRegression

from oemer.morph import morph_open
from oemer.logger import get_logger
from oemer.bbox import BBox


logger = get_logger(__name__)


class Grid:
    def __init__(self) -> None:
        self.id: Union[int, None] = None
        self.bbox: BBox = None  # type: ignore
        self.y_shift: int = 0

    @property
    def y_center(self) -> float:
        return (self.bbox[1]+self.bbox[3]) / 2

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


class GridGroup:
    def __init__(self) -> None:
        self.id: Union[int, None] = None
        self.reg_id: Union[int, None] = None
        self.bbox: BBox = None  # type: ignore
        self.gids: List[int] = []
        self.split_unit: int = None  # type: ignore

    @property
    def y_center(self):
        return round((self.bbox[1]+self.bbox[3]) / 2)

    def __lt__(self, tar: Self) -> bool:
        # Sort by width
        w = self.bbox[2] - self.bbox[0]
        tw = tar.bbox[2] - tar.bbox[0]
        return w < tw

    def __repr__(self):
        return f"Grid Group {self.id} / Width: {self.bbox[2]-self.bbox[0]} / BBox: {self.bbox}" \
            f" / Y-center: {self.y_center} / Reg. ID: {self.reg_id}"


def build_grid(st_pred: ndarray, split_unit: int = 11) -> Tuple[ndarray, List[Grid]]:
    grid_map = np.zeros(st_pred.shape) - 1
    h, w = st_pred.shape

    is_on = lambda data: np.sum(data) > split_unit//2

    grids: List[Grid] = []
    for i in range(0, w, split_unit):
        cur_y = 0
        last_y = 0
        cur_stat = is_on(st_pred[cur_y, i:i+split_unit])
        while cur_y < h:
            while cur_y < h and cur_stat == is_on(st_pred[cur_y, i:i+split_unit]):
                cur_y += 1
            if cur_stat and (cur_y-last_y < split_unit):
                # Switch off
                grid_map[last_y:cur_y, i:i+split_unit] = len(grids)
                gg = Grid()
                gg.bbox = (i, last_y, i+split_unit, cur_y)
                gg.id = len(grids)
                grids.append(gg)
            cur_stat = not cur_stat
            last_y = cur_y
    return grid_map, grids


def build_grid_group(grid_map: ndarray, grids: List[Grid]) -> Tuple[ndarray, List[GridGroup]]:
    regions, feat_num = scipy.ndimage.label(grid_map+1)
    grid_groups = []
    for i in range(feat_num):
        region = grid_map[regions==i+1]
        gids = list(np.unique(region).astype(int))
        gids = sorted(gids)
        lbox = grids[gids[0]].bbox
        rbox = grids[gids[-1]].bbox
        box = (
            min(lbox[0], rbox[0]),
            min(lbox[1], rbox[1]),
            max(lbox[2], rbox[2]),
            max(lbox[3], rbox[3]),
        )
        gg = GridGroup()
        gg.reg_id = i + 1
        gg.gids = gids
        gg.bbox = box
        gg.split_unit = lbox[2] - lbox[0]
        grid_groups.append(gg)

    grid_groups = sorted(grid_groups, reverse=True)
    gg_map = np.zeros_like(regions) - 1
    for idx, gg in enumerate(grid_groups):
        gg.id = idx
        gg_map[regions==gg.reg_id] = idx
        gg.reg_id = idx

    return gg_map, grid_groups


def connect_nearby_grid_group(
    gg_map: ndarray, 
    grid_groups: List[GridGroup], 
    grid_map: ndarray, 
    grids: List[Grid], 
    ref_count: int = 8, 
    max_step: int = 20
) -> ndarray:
    new_gg_map = np.copy(gg_map)
    ref_gids = grid_groups[0].gids[:ref_count]
    idx = 0
    gg = grid_groups[idx]
    remaining = set(range(len(grid_groups)))
    while remaining:
        # Check if not yet visited
        if gg.id not in remaining:
            if remaining:
                gid = remaining.pop()
                gg = grid_groups[gid]
                ref_gids = gg.gids[:ref_count]
            else:
                break
        else:
            remaining.remove(gg.id)

        if len(ref_gids) < 2:
            continue

        # Extend on the left side
        step_size = gg.split_unit
        centers = [grids[gid].y_center for gid in ref_gids]
        x = np.arange(len(centers)).reshape(-1, 1) * step_size
        model = LinearRegression().fit(x, centers)
        ref_box = grids[ref_gids[0]].bbox

        end_x = ref_box[0]
        h = ref_box[3] - ref_box[1]
        cands_box = []  # Potential trajectory
        for i in range(max_step):
            tar_x = (-i - 1) * step_size
            cen_y = model.predict([[tar_x]])[0]  # Interpolate y center
            y = int(round(cen_y - h / 2))
            region = new_gg_map[y:y+h, end_x-step_size:end_x]  # Area to check
            unique, counts = np.unique(region, return_counts=True)
            labels = set(unique)  # Overlapped grid group IDs
            if -1 in labels:
                labels.remove(-1)
                no_id_idx = np.where(unique==-1)[0][0]
                unique = np.delete(unique, no_id_idx)
                counts = np.delete(counts, no_id_idx)

            cands_box.append((end_x-step_size, y, end_x, y+h))
            if len(labels) == 0:
                end_x -= step_size
            else:
                cands_box = cands_box[:-1]  # Remove the overlapped box

                # Determine the overlapped grid group id
                if len(labels) > 1:
                    # Overlapped with more than one group
                    overlapped_size = sorted(zip(unique, counts), key=lambda it: it[1], reverse=True)
                    label = overlapped_size[0][0]
                else:
                    label = labels.pop()

                # Check the overlappiong with the traget grid group is valid.
                tar_box = grid_groups[label].bbox
                if tar_box[2] > end_x:
                    break

                # Start assign grid to disconnected position.
                # Get the grid ID.
                yidx, xidx = np.where(region==label)
                yidx += y
                xidx += end_x-step_size
                reg = grid_map[yidx, xidx]
                grid_id, counts = np.unique(reg, return_counts=True)
                if len(grid_id) > 1:
                    logger.warn(
                        "Detected multiple possible overlapping grids: %s. Reg. count: %s",
                        str(grid_id), str(counts))
                grid_id = int(grid_id[np.argmax(counts)])
                assert grid_id in grid_groups[label].gids, f"{grid_id}, {label}"
                grid = grids[grid_id]

                # Interpolate y centers between the start and end points again.
                centers = [grid.y_center, centers[0]]
                x = [-i-1, 0]  # type: ignore
                inter_func = interp1d(x, centers, kind='linear')

                # Start to insert grids between points
                cands_ids = []
                for bi, box in enumerate(cands_box):
                    interp_y = round(inter_func(-bi-1) - h/2)
                    grid = Grid()
                    box = (box[0], interp_y, box[2], interp_y+h)
                    grid.bbox = box
                    grid.id = len(grids)
                    cands_ids.append(len(grids))
                    gg.gids.append(len(grids))
                    gg.bbox = (
                        min(gg.bbox[0], box[0]),
                        min(gg.bbox[1], box[1]),
                        max(gg.bbox[2], box[2]),
                        max(gg.bbox[3], box[3])
                    )
                    gg.bbox = typing.cast(BBox, [int(bb) for bb in gg.bbox])
                    box = [int(bb) for bb in box]  # type: ignore
                    grids.append(grid)
                    new_gg_map[box[1]:box[3], box[0]:box[2]] = gg.id

                # Continue to track on the overlapped grid group.
                gg = grid_groups[label]
                gids = gg.gids + cands_ids[::-1]
                ref_gids = gids[:ref_count]

                break

    return new_gg_map


def build_mapping(gg_map: ndarray, min_width_ratio: float = 0.4) -> Tuple[ndarray, ndarray]:
    regions, num = scipy.ndimage.label(gg_map+1)
    min_width = gg_map.shape[1] * min_width_ratio

    points = []
    coords_y = np.zeros_like(gg_map)
    period = 10
    for i in range(num):
        y, x = np.where(regions==i+1)
        w = np.max(x) - np.min(x)
        if w < min_width:
            continue

        target_y = round(np.mean(y))

        uniq_x = np.unique(x)
        for ii, ux in enumerate(uniq_x):
            if ii % period == 0:
                meta_idx = np.where(x==ux)[0]
                sub_y = y[meta_idx]
                cen_y = round(np.mean(sub_y))
                coords_y[int(target_y), int(ux)] = cen_y
                points.append((target_y, ux))

    # Add corner case
    coords_y[0] = 0
    coords_y[-1] = len(coords_y) - 1
    for i in range(coords_y.shape[1]):
        points.append((0, i))
        points.append((coords_y.shape[0]-1, i))

    return coords_y, np.array(points)


def estimate_coords(staff_pred: ndarray) -> Tuple[ndarray, ndarray]:
    ker = np.ones((6, 1), dtype=np.uint8)
    pred = cv2.dilate(staff_pred.astype(np.uint8), ker)
    pred = morph_open(pred, (1, 6))

    logger.debug("Building grids")
    grid_map, grids = build_grid(pred)

    logger.debug("Labeling areas")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    logger.debug("Connecting lines")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    logger.debug("Building mapping")
    coords_y, points = build_mapping(new_gg_map)

    logger.debug("Dewarping")
    vals = coords_y[points[:, 0].astype(int), points[:, 1].astype(int)]
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1]
    coords_y = griddata(points, vals, (grid_x, grid_y), method='linear')

    coords_x = grid_y.astype(np.float32)
    coords_y = coords_y.astype(np.float32)
    return coords_x, coords_y


def dewarp(img: ndarray, coords_x: ndarray, coords_y: ndarray) -> ndarray:
    return cv2.remap(img.astype(np.float32), coords_x,coords_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


if __name__ == "__main__":
    f_name = "wind2"
    #f_name = "last"
    #f_name = "tabi"
    img_path = f"../test_imgs/{f_name}.jpg"

    img_path = "../test_imgs/Chihiro/7.jpg"
    #img_path = "../test_imgs/Gym/2.jpg"

    ori_img = cv2.imread(img_path)
    f_name, ext = os.path.splitext(os.path.basename(img_path))
    parent_dir = os.path.dirname(img_path)
    pkl_path = os.path.join(parent_dir, f_name+".pkl")
    ff = pickle.load(open(pkl_path, "rb"))
    st_pred = ff['staff']
    ori_img = cv2.resize(ori_img, (st_pred.shape[1], st_pred.shape[0]))

    ker = np.ones((6, 1), dtype=np.uint8)
    pred = cv2.dilate(st_pred.astype(np.uint8), ker)
    pred = morph_open(pred, (1, 6))

    print("Building grids")
    grid_map, grids = build_grid(pred)

    print("Labeling")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    print("Connecting")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    print("Estimating mapping")
    coords_y, points = build_mapping(new_gg_map)

    print("Dewarping")
    out = np.copy(ori_img)
    vals = coords_y[points[:, 0], points[:, 1]]
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1]
    mapping = griddata(points, vals, (grid_x, grid_y), method='linear')
    for i in range(out.shape[-1]):
        out[..., i] = cv2.remap(out[..., i].astype(np.float32), grid_y.astype(np.float32), mapping.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    mix = np.hstack([ori_img, out])


import random
def teaser():
    plt.clf()
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.subplot(231)
    plt.title("Predict")
    plt.axis('off')
    plt.imshow(st_pred, cmap="Greys")

    plt.subplot(232)
    plt.title("Morph")
    plt.axis('off')
    plt.imshow(pred, cmap='Greys')

    plt.subplot(233)
    plt.title("Quantize")
    plt.axis('off')
    plt.imshow(grid_map>0, cmap='Greys')

    plt.subplot(234)
    plt.title("Group")
    plt.axis('off')
    ggs = set(np.unique(gg_map))
    ggs.remove(-1)
    _gg_map = np.ones(gg_map.shape+(3,), dtype=np.uint8) * 255
    for i in ggs:
        ys, xs = np.where(gg_map==i)
        for c in range(3):
            v = random.randint(0, 255)
            _gg_map[ys, xs, c] = v
    plt.imshow(_gg_map)

    plt.subplot(235)
    plt.title("Connect")
    plt.axis('off')
    plt.imshow(new_gg_map>0, cmap='Greys')

    plt.subplot(236)
    plt.title("Dewarp")
    plt.axis('off')
    plt.imshow(out)

