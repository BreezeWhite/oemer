import enum
import pickle
import typing
from typing import List, Any,  Tuple, Union
from typing_extensions import Self

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import bool_, ndarray
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from oemer import layers
from oemer import exceptions as E
from oemer.logger import get_logger
from oemer.bbox import BBox, find_lines, get_bbox, get_center


logger = get_logger(__name__)


class LineLabel(enum.Enum):
    FIRST = 0
    SECOND = 1
    THIRD = 2
    FOURTH = 3
    FIFTH = 4


class Line:
    def __init__(self) -> None:
        self.points: Any = []
        self.label: Union[LineLabel, None] = None

    def add_point(self, y: int, x: int) -> None:
        self.points.append((y, x))
        self._y_center = None
        self._y_upper = None
        self._y_lower = None
        self._x_center = None
        self._x_left = None
        self._x_right = None
        self._slope = None

    @property
    def y_center(self) -> float:
        if not hasattr(self, "_y_center"):
            setattr(self, "_y_center", None)
        if self._y_center is not None:
            return self._y_center
        self._y_center = np.mean([point[0] for point in self.points])
        return typing.cast(float, self._y_center)

    @property
    def y_upper(self) -> Union[float, None]:
        if not hasattr(self, "_y_upper"):
            setattr(self, "_y_upper", None)
        if self._y_upper is not None:
            return self._y_upper
        self._y_upper = np.min([point[0] for point in self.points])
        return self._y_upper

    @property
    def y_lower(self) -> float:
        if not hasattr(self, "_y_lower"):
            setattr(self, "_y_lower", None)
        if self._y_lower is not None:
            return self._y_lower
        self._y_lower = np.max([point[0] for point in self.points])
        return typing.cast(float, self._y_lower)

    @property
    def x_center(self) -> float:
        if not hasattr(self, "_x_center"):
            setattr(self, "_x_center", None)
        if self._x_center is not None:
            return self._x_center
        self._x_center = np.mean([point[1] for point in self.points])
        return typing.cast(float, self._x_center)

    @property
    def x_left(self) -> float:
        if not hasattr(self, "_x_left"):
            setattr(self, "_x_left", None)
        if self._x_left is not None:
            return self._x_left
        self._x_left = np.min([point[1] for point in self.points])
        return typing.cast(float, self._x_left)  

    @property
    def x_right(self) -> float:
        if not hasattr(self, "_x_right"):
            setattr(self, "_x_right", None)
        if self._x_right is not None:
            return self._x_right
        self._x_right = np.max([point[1] for point in self.points])
        return typing.cast(float, self._x_right)

    @property
    def slope(self) -> float:
        if not hasattr(self, "_slope"):
            self._slope = None
        if self._slope is not None:
            return self._slope
        points = np.array(self.points)
        ys = [p[0] for p in points]
        xs = [[p[1]] for p in points]
        model = LinearRegression()
        model.fit(xs, ys)
        self._slope = model.coef_[0]
        return typing.cast(float, self._slope)

    def __lt__(self, line: Self) -> bool:
        return self.y_center < line.y_center

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return "Line(\n" \
            f"\tPoint count: {len(self.points)}\n" \
            f"\tCenter: {self.y_center}\n" \
            f"\tUpper bound: {self.y_upper}\n" \
            f"\tLower bound: {self.y_lower}\n" \
            f"\tLabel: {self.label}\n" \
            f"\tSlope: {self.slope}\n" \
            ")\n"


class Staff:
    def __init__(self) -> None:
        self.lines: List[Line] = []
        self.track: Union[int, None] = None
        self.group: Union[int, None]  = None
        self.is_interp: bool = False

    def add_line(self, line: Line) -> None:
        self.lines.append(line)
        self._y_center: Union[float, None] = None
        self._y_upper = None
        self._y_lower = None
        self._x_center: Union[float, None] = None
        self._x_left = None
        self._x_right = None
        self._unit_size: Union[float, None] = None
        self._slope: Union[float, None] = None

    @property
    def y_center(self) -> float:
        if not hasattr(self, "_y_center"):
            setattr(self, "_y_center", None)
        if self._y_center is not None:
            return self._y_center
        self._y_center = float(np.mean([line.y_center for line in self.lines]))
        return self._y_center

    @y_center.setter
    def y_center(self, val):
        self._y_center = val

    @property
    def y_upper(self) -> float:
        if not hasattr(self, "_y_upper"):
            setattr(self, "_y_upper", None)
        if self._y_upper is not None:
            return self._y_upper
        self._y_upper = np.min([line.y_upper for line in self.lines])
        return typing.cast(float, self._y_upper)

    @y_upper.setter
    def y_upper(self, val):
        self._y_upper = val

    @property
    def y_lower(self) -> float:
        if not hasattr(self, "_y_lower"):
            setattr(self, "_y_lower", None)
        if self._y_lower is not None:
            return self._y_lower
        self._y_lower = np.max([line.y_lower for line in self.lines])
        return typing.cast(float, self._y_lower)

    @y_lower.setter
    def y_lower(self, val):
        self._y_lower = val

    @property
    def x_center(self) -> float:
        if not hasattr(self, "_x_center"):
            setattr(self, "_x_center", None)
        if self._x_center is not None:
            return self._x_center
        self._x_center = float(np.mean([line.x_center for line in self.lines]))
        return self._x_center

    @x_center.setter
    def x_center(self, val):
        self._x_center = val

    @property
    def x_left(self) -> float:
        if not hasattr(self, "_x_left"):
            setattr(self, "_x_left", None)
        if self._x_left is not None:
            return self._x_left
        self._x_left = np.min([line.x_left for line in self.lines])
        return typing.cast(float, self._x_left)

    @x_left.setter
    def x_left(self, val):
        self._x_left = val

    @property
    def x_right(self) -> float:
        if not hasattr(self, "_x_right"):
            setattr(self, "_x_right", None)
        if self._x_right is not None:
            return self._x_right
        self._x_right = np.max([line.x_right for line in self.lines])
        return typing.cast(float, self._x_right)

    @x_right.setter
    def x_right(self, val):
        self._x_right = val

    @property
    def unit_size(self) -> float:
        # The very basic size for measuring all the symbols.
        if not hasattr(self, "_unit_size"):
            setattr(self, "_unit_size", None)
        if self._unit_size is not None:
            return self._unit_size
        centers = [line.y_center for line in self.lines]
        gaps = [centers[i] - centers[i-1] for i in range(1, len(self.lines))]
        self._unit_size = float(np.mean(gaps))
        return self._unit_size

    @property
    def incomplete(self) -> bool:
        return len(self.lines) != 5

    @property
    def slope(self) -> float:
        if not hasattr(self, "_slope"):
            self._slope = None
        if self._slope is not None:
            return self._slope
        self._slope = float(np.mean([l.slope for l in self.lines]))
        return self._slope

    def duplicate(self, x_offset=0, y_offset=0):
        st = Staff()
        for line in self.lines:
            new_l = Line()
            for y, x in line.points:
                new_l.add_point(y+y_offset, x+x_offset)
            st.add_line(new_l)
        return st

    def __lt__(self, st):
        return self.y_center < st.y_center

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        return "Staff(\n" \
            f"\tLines: {len(self.lines)}\n" \
            f"\tCenter: {self.y_center}\n" \
            f"\tUpper bound: {self.y_upper}\n" \
            f"\tLower bound: {self.y_lower}\n" \
            f"\tUnit size: {self.unit_size}\n" \
            f"\tTrack: {self.track}\n" \
            f"\tGroup: {self.group}\n" \
            f"\tIs interpolation: {self.is_interp}\n" \
            f"\tSlope: {self.slope}\n" \
            ")\n"

    def __sub__(self, st: List[int]) -> float:
        if isinstance(st, Staff):
            x, y = st.x_center, st.y_center
        else:
            x, y = st
        x_dist = (x - self.x_center) ** 2
        y_dist = (y - self.y_center) ** 2
        return (x_dist + y_dist) ** 0.5


def init_zones(staff_pred: ndarray, splits: int) -> Tuple[ndarray, int, int, int]:
    ys, xs = np.where(staff_pred > 0)

    # Define left and right bound
    accum_x = np.sum(staff_pred, axis=0)
    accum_x = accum_x / np.mean(accum_x)
    half = round(len(accum_x) / 2)
    right_bound = min(max(xs) + 50, staff_pred.shape[1])
    left_bound = max(min(xs) - 50, 0)
    for i in range(half+10, len(accum_x)):
        if np.mean(accum_x[i-10:i]) < 0.1:
            right_bound = i
            break
    for i in range(half-10, 0, -1):
        if np.mean(accum_x[i:i+10]) < 0.1:
            left_bound = i
            break

    bottom_bound = min(max(ys) + 100, len(staff_pred))
    step_size = round((right_bound - left_bound) / splits)
    zones = []
    for start in range(left_bound, right_bound, step_size):
        end = start + step_size
        if right_bound - end < step_size:
            end = right_bound
            zones.append(range(start, end))
            break
        zones.append(range(start, end))
    return np.array(zones, dtype=object), left_bound, right_bound, bottom_bound


def extract(
    splits: int = 8, 
    line_threshold: float = 0.8, 
    horizontal_diff_th: float = 0.1, 
    unit_size_diff_th: float = 0.1, 
    barline_min_degree: int = 75
) -> Tuple[ndarray, ndarray]:
    # Fetch parameters from layers
    staff_pred = layers.get_layer('staff_pred')

    # Start process
    zones, *_ = init_zones(staff_pred, splits=splits)
    all_staffs: List[List[Staff]] = []
    for rr in zones:
        print(rr[0], rr[-1], end=' ')
        rr = np.array(rr, dtype=np.int64)
        staffs = extract_part(staff_pred[:, rr], x_offset=rr[0], line_threshold=line_threshold)
        if staffs is not None:
            all_staffs.append(staffs)
            print(len(staffs))
    aligned_staffs: np.ndarray = align_staffs(all_staffs)

    # Use barline information to infer the number of tracks for each group.
    num_track = further_infer_track_nums(aligned_staffs, min_degree=barline_min_degree)
    logger.debug(f"Tracks: {num_track}")
    for col_sts in aligned_staffs:
        for idx, st in enumerate(col_sts):
            st.track = idx % num_track
            st.group = idx // num_track

    # Validate staffs across zones.
    # Should have same number of staffs
    if not all([len(staff) == len(aligned_staffs[0]) for staff in aligned_staffs]):
        raise Exception
    assert all([len(staff) == len(aligned_staffs[0]) for staff in aligned_staffs])

    norm = lambda data: np.abs(np.array(data) / np.mean(data) - 1)
    valid_staffs: list[list[Staff]] = []
    for staffs in aligned_staffs.T:
        # Should all have 5 lines
        line_num = [len(staff.lines) for staff in staffs]
        if len(set(line_num)) != 1:
            print(f"Some of the stafflines contains less or more than 5 lines: {line_num}")
            continue

        # Check Staffs that are approximately at the same row.
        centers = np.array([staff.y_center for staff in staffs])
        if not np.all(norm(centers) < horizontal_diff_th):
            print(f"Centers of staff parts at the same row not aligned (Th: {horizontal_diff_th}): {norm(centers)}")
            continue

        # Unit sizes should roughly all the same
        unit_size = np.array([staff.unit_size for staff in staffs])
        if not np.all(norm(unit_size) < unit_size_diff_th):          
            print(f"Unit sizes not consistent (th: {unit_size_diff_th}): {norm(unit_size)}")
            continue
        valid_staffs.append(staffs)

    return np.array(valid_staffs).T, zones


def extract_part(pred: ndarray, x_offset: int, line_threshold: float = 0.8) -> List[Staff]:
    # Extract lines
    lines, _ = extract_line(pred, x_offset=x_offset, line_threshold=line_threshold)

    # To assure there contains at leat one staff lines and above
    if len(lines) < 5:
        return None  # type: ignore

    staffs = []
    line_buffer: Any = []
    for idx, line in enumerate(lines):
        lid = idx % 5
        assert line.label == LineLabel(lid), f"{line}, {lid}, {idx}"
        if lid == 0 and line_buffer:
            staff = Staff()
            assert len(line_buffer) == 5, len(line_buffer)
            for l in line_buffer:
                staff.add_line(l)
            staffs.append(staff)
            line_buffer = []
        line_buffer.append(line)

    # Clear out line buffer
    staff = Staff()
    for l in line_buffer:
        staff.add_line(l)
    staffs.append(staff)

    return staffs


def extract_line(pred: ndarray, x_offset: int, line_threshold: float = 0.8) -> Tuple[ndarray, ndarray]:
    # Split into zones horizontally and detects staff lines separately.
    count = np.zeros(len(pred), dtype=np.uint16)
    sub_ys, sub_xs = np.where(pred > 0)
    for y in sub_ys:
        count[y] += 1

    # Find peaks
    count = np.insert(count, [0, len(count)], [0, 0])  # Prepend / append
    norm = (count - np.mean(count)) / np.std(count)
    centers, _ = find_peaks(norm, height=line_threshold, distance=8, prominence=1)
    centers -= 1
    norm = norm[1:-1] # Remove prepend / append
    valid_centers, groups = filter_line_peaks(centers, norm)

    # Assign points to each staff line
    cc = centers[valid_centers]
    max_gap = np.mean(np.sort(cc[1:] - cc[:-1])[:3])
    lines = [Line() for _ in range(len(centers))]
    for y, x in zip(sub_ys, sub_xs):
        closest_cen = np.argmin(np.abs(centers - y))
        cen = centers[closest_cen]
        if valid_centers[closest_cen] \
                and (norm[y] > min(line_threshold, 1.2)) \
                and (abs(y - cen) < max_gap):
            lines[closest_cen].add_point(y, x+x_offset)

    # Assign labels
    last_group = groups[0]
    cur_line_id = 0
    idx = 0
    pack = sorted(zip(lines, valid_centers, groups), key=lambda obj: obj[0])
    for line, valid, grp in pack:
        if not valid:
            continue
        if grp != last_group:
            cur_line_id = 0
            last_group = grp

        line.label = LineLabel(cur_line_id)
        cur_line_id += 1
        idx += 1

    lines = np.array(lines)[valid_centers]
    return lines, norm  # type: ignore


def filter_line_peaks(peaks: ndarray, norm: ndarray, max_gap_ratio: float = 1.5) -> Tuple[ndarray, List[int]]:
    valid_peaks = np.array([True for _ in range(len(peaks))])

    # Filter by height
    for idx, p in enumerate(peaks):
        if norm[p] > 15:
            valid_peaks[idx] = False

    # Filter by x-axis
    gaps = peaks[1:] - peaks[:-1]
    count = max(5, round(len(peaks) * 0.2))
    approx_unit = np.mean(np.sort(gaps)[:count])
    max_gap = approx_unit * max_gap_ratio

    ext_peaks = [peaks[0]-max_gap-1] + list(peaks)  # Prepend an invalid peak for better handling edge case
    groups = []
    group = -1
    for i in range(1, len(ext_peaks)):
        if ext_peaks[i] - ext_peaks[i-1] > max_gap:
            group += 1
        groups.append(group)

    # print(f"Max gap: {max_gap}, Approx unit: {approx_unit}")
    # print(f"Gaps: {gaps}")
    # print(f"Groups: {groups}")

    groups.append(groups[-1]+1)  # Append an invalid group for better handling edge case
    cur_g = groups[0]
    count = 1
    for idx in range(1, len(groups)):
        group = groups[idx]
        if group == cur_g:
            count += 1
            continue

        if count < 5:
            # Incomplete peaks. Also eliminates the top and bottom incomplete staff lines.
            valid_peaks[idx-count:idx] = False
        elif count > 5:
            cand_peaks = peaks[idx-count:idx]
            head_part = cand_peaks[:5]
            tail_part = cand_peaks[-5:]
            if sum(norm[head_part]) > sum(norm[tail_part]):
                valid_peaks[idx-count+5:idx] = False
            else:
                valid_peaks[idx-count:idx-5] = False

        cur_g = group
        count = 1
    return valid_peaks, groups[:-1]


def align_staffs(staffs: List[List[Staff]], max_dist_ratio: int = 3) -> ndarray:
    len_types = set(len(st_part) for st_part in staffs)
    if len(len_types) == 1:
        # All columns contains the same amount of sub-staffs
        return np.array(staffs)

    # Fill sub-staffs into the grid
    max_len = max(len_types)
    grid = np.zeros((len(staffs), max_len), dtype=object)
    for idx, st_part in enumerate(staffs):
        if len(st_part) == max_len:
            grid[idx] = np.array(st_part)

    # Define utility function for searching nearby sub-staffs
    def get_nearby_sts(j, row):
        dists = [(idx, abs(idx-j)) for idx in range(len(row))]
        dists = sorted(dists, key=lambda it: it[1])
        idxs = [it[0] for it in dists]

        nearby_sts = []
        for near_idx in idxs:
            if isinstance(row[near_idx], Staff):
                nearby_sts.append((near_idx, row[near_idx]))
            if len(nearby_sts) >= 2:
                break
        return nearby_sts

    def get_nearest_ori_st(ref_st, ori_st_col):
        max_dist = ref_st.unit_size * max_dist_ratio
        for st in ori_st_col:
            dist = abs(st.y_center - ref_st.y_center)
            if dist < max_dist:
                return st
        return None

    # Start to fill in the empty sub-staffs
    for i in range(max_len):
        row = grid[:, i]
        for j, obj in enumerate(row):
            if isinstance(obj, Staff):
                continue

            ori_st_part = staffs[j]
            sts = get_nearby_sts(j, row)
            assert len(ori_st_part) < max_len, f"{len(ori_st_part)}, {max_len}"
            assert len(sts) > 0, sts

            # Check there is an original extraction result that is within
            # the same row. Otherwise interpolate a new one and insert to
            # the extraction result.
            ori_st = get_nearest_ori_st(sts[0][1], ori_st_part)
            if ori_st is not None:
                grid[j, i] = ori_st
                continue

            # Interpolate one new sub-staff.
            if len(sts) == 1:
                ref_idx, ref_st = sts[0]
                assert ref_idx != j
                width = ref_st.x_right - ref_st.x_left
                x_offset = width * (j - ref_idx)
                new_st = ref_st.duplicate(x_offset=x_offset)
            else:
                assert len(sts) == 2, sts
                (idx1, ref1), (idx2, ref2) = sts

                assert idx1 != j, idx1
                assert idx2 != j, idx2
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                    ref1, ref2 = ref2, ref1
                if j < idx1:
                    # Left outer interpolation
                    r1, r2 = idx1-j, idx2-idx1
                    x_offset = -(ref2.x_center-ref1.x_center) * (r1 / r2)
                    y_offset = -(ref2.y_center-ref1.y_center) * (r1 / r2)
                    new_st = ref1.duplicate(x_offset=x_offset, y_offset=y_offset)
                elif idx1 <= j < idx2:
                    # Inner interpolation
                    r1, r2 = j-idx1, idx2-idx1
                    x_offset = (ref2.x_center-ref1.x_center) * (r1 / r2)
                    y_offset = (ref2.y_center-ref1.y_center) * (r1 / r2)
                    new_st = ref1.duplicate(x_offset=x_offset, y_offset=y_offset)
                else:
                    # Right outer interpolation
                    r1, r2 = idx2-idx1, j-idx2
                    x_offset = (ref2.x_center-ref1.x_center) * (r2 / r1)
                    y_offset = (ref2.y_center-ref1.y_center) * (r2 / r1)
                    new_st = ref2.duplicate(x_offset=x_offset, y_offset=y_offset)
            new_st.is_interp = True
            grid[j, i] = new_st
    return grid


def further_infer_track_nums(staffs: ndarray, min_degree: int = 75) -> int:
    # Fetch parameters
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    clefs = layers.get_layer('clefs_keys_pred')

    mix = symbols - stems - notehead - clefs
    mix[mix<0] = 0

    # Find straight lines in the symbols
    lines = find_lines(mix)
    lines = filter_lines(lines, staffs, min_degree=min_degree)
    bmap = get_barline_map(symbols, lines) + stems
    bmap[bmap>1] = 1

    # Morph possible disconnections
    ker = np.ones((5, 2), dtype=np.uint8)
    ext_bmap = cv2.erode(cv2.dilate(bmap.astype(np.uint8), ker), ker)
    bboxes = get_bbox(ext_bmap)

    # Height to unit size ratio of barline candidates
    h_ratios = []
    for box in bboxes:
        h = box[3] - box[1]
        unit_size = naive_get_unit_size(staffs, *get_center(box))
        if h > unit_size:
            h_ratios.append(h / unit_size)
    h_ratios = np.array(h_ratios)  # type: ignore

    num_track = 1
    factor = 10
    for i in range(1, 10):
        valid_h = len(h_ratios[h_ratios>factor*i])  # type: ignore
        if valid_h * (i+1) > staffs.shape[1]:
            num_track += 1
        else:
            break
    return num_track


def get_degree(line: BBox) -> float:
    return np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0]))


def filter_lines(lines: List[BBox], staffs: ndarray, min_degree: int = 75) -> List[BBox]:
    min_y = 9999999
    min_x = 9999999
    max_y = 0
    max_x = 0
    for st in staffs.reshape(-1, 1).squeeze():
        min_y = min(min_y, st.y_upper)
        min_x = min(min_x, st.x_left)
        max_y = max(max_y, st.y_lower)
        max_x = max(max_x, st.x_right)

    cands = []
    for line in lines:
        # Check angle
        degree = get_degree(line)
        if degree < min_degree:
            continue

        # Check position
        if line[1] < min_y \
                or line[3] > max_y \
                or line[0] < min_x \
                or line[2] > max_x:
            continue

        cands.append(line)
    return cands


def get_barline_map(symbols: ndarray, bboxes: List[BBox]) -> ndarray:
    img = np.zeros_like(symbols)
    for box in bboxes:
        box = list(box)  # type: ignore
        if box[2]-box[0] == 0:
            box[2] += 1  # type: ignore
        img[box[1]:box[3], box[0]:box[2]] += symbols[box[1]:box[3], box[0]:box[2]]
    img[img>1] = 1
    return img


def naive_get_unit_size(staffs: ndarray, x: int, y: int) -> float:
    flat_staffs = staffs.reshape(-1, 1).squeeze()

    def dist(st: Staff) -> float:
        x_diff = st.x_center - x
        y_diff = st.y_center - y
        return x_diff ** 2 + y_diff ** 2  # type: ignore

    dists = [(st.unit_size, dist(st)) for st in flat_staffs]
    dists = sorted(dists, key=lambda it: it[1])
    return dists[0][0]


if __name__ == "__main__":
    f_name = "last"
    f_name = "tabi"
    #f_name = "tabi_page2"
    #f_name = "PXL2"
    #f_name = "girl"
    #f_name = "1"

    pred = pickle.load(open(f"../test_imgs/{f_name}.pkl", "rb"))['staff']
    layers.register_layer("staff_pred", pred)
    rr = range(1130, 1400)
    #staffs, zones = extract()
    #staffs = extract_part(pred[..., rr], 0)
    lines, norm = extract_line(pred[..., rr], 0)

    data = pred[..., rr]
    count = np.zeros(len(data))
    ys, xs = np.where(data>0)
    for y in ys:
        count[y] += 1
    count = np.array([0] + list(count))
    norm = (count - np.mean(count)) / np.std(count)

    threshold = 0.8
    peaks, _ = find_peaks(norm, height=threshold, distance=8, prominence=1)
    valid_peaks, groups = filter_line_peaks(peaks, norm)
    peaks = peaks[valid_peaks]
    plt.plot(norm)  # type: ignore
    plt.plot(peaks, [threshold]*len(peaks), 'ro')
    plt.show()
