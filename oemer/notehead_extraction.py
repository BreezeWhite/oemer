import enum
import typing
from typing import List, Tuple, Any, Union

import cv2
import scipy.ndimage
import numpy as np
from numpy import ndarray

from oemer import layers
from oemer.constant import NoteHeadConstant as nhc
from oemer.bbox import BBox, get_bbox, get_center, merge_nearby_bbox, rm_merge_overlap_bbox, to_rgb_img
from oemer.utils import get_unit_size, find_closest_staffs, get_global_unit_size
from oemer.logger import get_logger
from oemer.staffline_extraction import Staff

# Globals
nn_img: ndarray

logger = get_logger(__name__)


class NoteType(enum.Enum):
    WHOLE = 0
    HALF = 1
    QUARTER = 2
    EIGHTH = 3
    SIXTEENTH = 4
    THIRTY_SECOND = 5
    SIXTY_FOURTH = 6
    TRIPLET = 7
    OTHERS = 8

    # An intermediate product while parsing.
    HALF_OR_WHOLE = 9


class NoteHead:
    def __init__(self) -> None:
        self.points: List[tuple] = []
        self.pitch: Union[int, None] = None
        self.has_dot: bool = False
        self.bbox: BBox = None  # type: ignore 
        self.stem_up: Union[bool, None] = None
        self.stem_right: Union[bool, None] = None
        self.track: Union[int, None] = None
        self.group: Union[int, None] = None
        self.staff_line_pos: int = None  # type: ignore
        self.invalid: Union[bool, None] = False  # May be false positive
        self.id: Union[int, None] = None
        self.note_group_id: Union[int, None] = None
        self.sfn: Any = None  # See symbols_extraction.py

        # Protected attributes
        self._label: Union[NoteType, None] = None

    @property
    def label(self) -> Union[NoteType, None]:
        if self.invalid:
            logger.warning(f"Note {self.id} is not a valid note.")
            return None
        return self._label

    @label.setter
    def label(self, label: NoteType):
        if self._label is not None:
            logger.debug(f"The label has been set to: {self._label}."
                          " Use 'force_set_label' instead if you really want to modify the label.")
            return
        self._label = label

    def force_set_label(self, label: NoteType):
        logger.debug(f"Force set label from {self.label} to {label}")
        assert isinstance(label, NoteType)
        self._label = label

    def add_point(self, x: int, y: int) -> None:
        self.points.append((y, x))

    def __lt__(self, nt):
        # Compare by position
        return self.staff_line_pos < nt.staff_line_pos

    def __repr__(self):
        return f"Notehead {self.id}(\n" \
            f"\tPoints: {len(self.points)}\n" \
            f"\tBounding box: {self.bbox}\n" \
            f"\tStem up: {self.stem_up}\n" \
            f"\tTrack: {self.track}\n" \
            f"\tGroup: {self.group}\n" \
            f"\tPitch: {self.pitch}\n" \
            f"\tDot: {self.has_dot}\n" \
            f"\tLabel: {self.label}\n" \
            f"\tStaff line pos: {self.staff_line_pos}\n" \
            f"\tIs valid: {not self.invalid}\n" \
            f"\tNote group ID: {self.note_group_id}\n" \
            f"\tSharp/Flat/Natural: {self.sfn}\n" \
            f")\n"


def morph_notehead(pred: ndarray, unit_size: float) -> ndarray:
    small_size = int(round(unit_size / 3))
    small_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_size, small_size))
    pred = cv2.erode(cv2.dilate(pred.astype(np.uint8), small_ker), small_ker)
    size = (
        int(round(unit_size*nhc.NOTEHEAD_MORPH_WIDTH_FACTOR)),
        int(round(unit_size*nhc.NOTEHEAD_MORPH_HEIGHT_FACTOR))
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    img = cv2.erode(pred.astype(np.uint8), kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size[0]+1, size[1]+1))
    return cv2.dilate(img, kernel)


def adjust_bbox(bbox, noteheads):
    region = noteheads[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    ys, _ = np.where(region>0)
    if len(ys) == 0:
        # Invalid note. Will be eliminated with zero height.
        return None
    top = np.min(ys) + bbox[1] - 1
    bottom = np.max(ys) + bbox[1] + 1
    return (bbox[0], top, bbox[2], bottom)


def check_bbox_size(bbox: BBox, noteheads: ndarray, unit_size: float) -> List[BBox]:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cen_x, _ = get_center(bbox)
    note_w = nhc.NOTEHEAD_SIZE_RATIO * unit_size
    note_h = unit_size

    new_bbox = []
    if abs(w - note_w) > abs(w - note_w*2):
        # Contains at least two notes, one left and one right.
        left_box = (bbox[0], bbox[1], cen_x, bbox[3])
        right_box = (cen_x, bbox[1], bbox[2], bbox[3])

        # Upper and lower bounds could have changed
        left_box = adjust_bbox(left_box, noteheads)
        right_box = adjust_bbox(right_box, noteheads)

        # Check recursively
        if left_box is not None:
            new_bbox.extend(check_bbox_size(left_box, noteheads, unit_size))
        if right_box is not None:
            new_bbox.extend(check_bbox_size(right_box, noteheads, unit_size))

    # Check height
    if len(new_bbox) > 0:
        tmp_new = []
        for box in new_bbox:
            tmp_new.extend(check_bbox_size(box, noteheads, unit_size))
        new_bbox = tmp_new
    else:
        num_notes = int(round(h / note_h))
        if num_notes > 0:
            sub_h = h // num_notes
            for i in range(num_notes):
                sub_box = (
                    bbox[0],
                    round(bbox[1] + i*sub_h),
                    bbox[2],
                    round(bbox[1] + (i+1)*sub_h)
                )
                new_bbox.append(sub_box)

    return new_bbox


def filter_notehead_bbox(
    bboxes: List[BBox],
    notehead: ndarray,
    min_h_ratio: float = 0.4,
    max_h_ratio: int = 5,
    min_w_ratio: float = 0.3,
    max_w_ratio: int = 3,
    min_area_ratio: float = 0.5) -> List[BBox]:

    # Fetch parameters
    zones = layers.get_layer('zones')

    # Start process
    # Get the left and right bound.
    min_x = zones[0][0]
    max_x = zones[-1][-1]

    valid_bboxes = []
    for bbox in bboxes:
        cen_x, cen_y = get_center(bbox)
        unit_size = get_unit_size(cen_x, cen_y)

        # Check x-axis
        if (cen_x < min_x + nhc.CLEF_ZONE_WIDTH_UNIT_RATIO*unit_size) or (cen_x > max_x):
            continue

        # Check size
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        if (h < unit_size*min_h_ratio) or (h > unit_size*max_h_ratio):
            continue
        if (w < unit_size*min_w_ratio*nhc.NOTEHEAD_SIZE_RATIO) \
                or (w > unit_size*max_w_ratio*nhc.NOTEHEAD_SIZE_RATIO):
            continue

        # Check area size
        region = notehead[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        min_count = h * w * min_area_ratio
        count = region[region>0].size
        if count < min_count:
            continue

        valid_bboxes.append(bbox)
    return valid_bboxes


def get_notehead_bbox(
    pred: ndarray,
    global_unit_size: float,
    min_h_ratio: float = 0.4,
    max_h_ratio: int = 5,
    min_w_ratio: float = 0.3,
    max_w_ratio: int = 3,
    min_area_ratio: float = 0.6
) -> List[BBox]:
    logger.debug("Morph noteheads")
    note = morph_notehead(pred, unit_size=global_unit_size)
    bboxes = get_bbox(note)
    bboxes = rm_merge_overlap_bbox(bboxes)
    result_bboxes: List[BBox] = []
    for box in bboxes:
        unit_size = get_unit_size(*get_center(box))
        checked_boxes = check_bbox_size(box, pred, unit_size)  # type: ignore
        result_bboxes.extend(checked_boxes)
    logger.debug("Detected noteheads: %d", len(result_bboxes))

    logger.debug("Filtering noteheads")
    bboxes = filter_notehead_bbox(
        result_bboxes,
        note,
        min_h_ratio=min_h_ratio,
        max_h_ratio=max_h_ratio,
        min_w_ratio=min_w_ratio,
        max_w_ratio=max_w_ratio,
        min_area_ratio=min_area_ratio
    )
    logger.debug("Detected noteheads after filtering: %d", len(bboxes))
    return bboxes


def fill_hole(region: ndarray) -> ndarray:
    tar = np.copy(region)

    h, w = tar.shape

    # Scan by row
    for yi in range(h):
        cur = 0
        cand_y = []
        cand_x = []
        while cur < w:
            if tar[yi, cur] > 0:
                break
            cur += 1
        while cur < w:
            if tar[yi, cur] == 0:
                break
            cur += 1
        while cur < w:
            if tar[yi, cur] > 0:
                break
            cand_y.append(yi)
            cand_x.append(cur)
            cur += 1
        if cur < w:
            cand_y = np.array(cand_y)  # type: ignore
            cand_x = np.array(cand_x)  # type: ignore
            tar[cand_y, cand_x] = 1

    # Scan by column
        for xi in range(w):
            cur = 0
            cand_y = []
            cand_x = []
            while cur < h:
                if tar[cur, xi] > 0:
                    break
                cur += 1
            while cur < h:
                if tar[cur, xi] == 0:
                    break
                cur += 1
            while cur < h:
                if tar[cur, xi] > 0:
                    break
                cand_y.append(cur)
                cand_x.append(xi)
                cur += 1
            if cur < h:
                cand_y = np.array(cand_y)  # type: ignore
                cand_x = np.array(cand_x)  # type: ignore
                tar[cand_y, cand_x] = 1
    return tar


def gen_notes(bboxes: List[ndarray], symbols: ndarray) -> List[NoteHead]:
    notes = []
    for bbox in bboxes:
        # Instantiate notehead.
        nn = NoteHead()
        nn.bbox = typing.cast(BBox, bbox)

        # Add points
        region = symbols[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ys, xs = np.where(region > 0)
        ys += bbox[1]
        xs += bbox[0]
        for y, x in zip(ys, xs):
            nn.add_point(x, y)

        # Assign group and track to the note
        def assign_group_track(st: Staff) -> None:
            nn.group = st.group
            nn.track = st.track

        cen_x, cen_y = get_center(bbox)
        st1, st2 = find_closest_staffs(cen_x, cen_y)
        if (st1.y_center == st2.y_center) \
                or (st1.y_upper <= cen_y <= st1.y_lower):
            assign_group_track(st1)
            st_master = st1
        else:
            up_st, lo_st = (st1, st2) if st1.y_center < st2.y_center else (st2, st1)
            sts_cen = (up_st.y_center + lo_st.y_center) / 2
            if cen_y < sts_cen:
                assign_group_track(up_st)
                st_master = up_st
            else:
                assign_group_track(lo_st)
                st_master = lo_st

        # Determine staffline position. Notice that this doesn't equal to pitch.
        # The value could also be negative. The zero index starts from the position
        # same as D4, assert the staffline is in treble clef. The value increases
        # as the pitch goes up.
        # Build centers of each position first.
        step = st_master.unit_size / 2
        pos_cen = [l.y_center for l in st_master.lines[::-1]]
        tmp_inter = []
        for idx, cen in enumerate(pos_cen[:-1]):
            interp = (cen + pos_cen[idx+1]) / 2
            tmp_inter.append(interp)
        for idx, interp in enumerate(tmp_inter):
            pos_cen.insert(idx*2+1, interp)
        pos_cen = [pos_cen[0]+step] + pos_cen + [pos_cen[-1]-step]

        # Estimate position by the closest center.
        pos_idx = np.argmin(np.abs(np.array(pos_cen)-cen_y))
        if 0 < pos_idx < len(pos_cen)-1:
            nn.staff_line_pos = int(pos_idx)
        elif pos_idx == 0:
            diff = abs(pos_cen[0] - cen_y)
            pos = round(diff / step)
            nn.staff_line_pos = -pos
        else:
            diff = abs(pos_cen[-1] - cen_y)
            pos = round(diff / step) + len(pos_cen) - 1
            nn.staff_line_pos = pos

        notes.append(nn)
    return notes


def parse_stem_info(notes: List[NoteHead]) -> None:
    # Fetch parameters
    stems = layers.get_layer('stems_rests_pred')

    ker = np.ones((3, 2), dtype=np.uint8)
    enhanced_stems = cv2.dilate(stems.astype(np.uint8), ker)
    st_map, _ = scipy.ndimage.label(enhanced_stems)

    for note in notes:
        box = note.bbox
        region = st_map[box[1]:box[3], box[0]:box[2]]  # type: ignore
        lls = set(np.unique(region))
        if 0 in lls:
            lls.remove(0)
        if len(lls) == 0:
            continue

        label = lls.pop()
        _, xi = np.where(st_map==label)
        st_cen_x = np.mean(xi)
        cen_x = (box[0] + box[2]) / 2
        on_right = st_cen_x > cen_x
        note.stem_right = bool(on_right)
        # start_y = box[1] - offset
        # end_y = box[3] + offset
        # left_sum = np.sum(stems[start_y:end_y, box[0]-offset:box[2]])
        # right_sum = np.sum(stems[start_y:end_y, box[0]:box[2]+offset])
        # if left_sum == 0 and right_sum == 0:
        #     continue
        # elif left_sum > right_sum:
        #     note.stem_right = False
        # else:
        #     note.stem_right = True

def extract(
    min_h_ratio: float = 0.4,
    max_h_ratio: int = 5,
    min_w_ratio: float = 0.3,
    max_w_ratio: int = 3,
    min_area_ratio: float = 0.5,
    max_whole_note_width_factor: float = 1.5,
    y_dist_factor: int = 5,
    hollow_filled_ratio_th: float = 1.3
) -> List[NoteHead]:

    # Fetch parameters from layers
    pred = layers.get_layer('notehead_pred')
    symbols = layers.get_layer('symbols_pred')

    unit_size = get_global_unit_size()
    logger.info("Analyzing notehead bboxes")
    bboxes = get_notehead_bbox(
        pred,
        unit_size,
        min_h_ratio=min_h_ratio,
        max_h_ratio=max_h_ratio,
        min_w_ratio=min_w_ratio,
        max_w_ratio=max_w_ratio,
        min_area_ratio=min_area_ratio
    )

    global nn_img
    nn_img = to_rgb_img(pred)

    ## -- Special cases for half/whole notes -- ##
    # Whole notes has wider width, and may be splited into two bboxes
    merged_box = merge_nearby_bbox(bboxes, distance=unit_size*max_whole_note_width_factor, y_factor=y_dist_factor)
    solid_box = []
    hollow_box = []
    for box in merged_box:
        # Fix shifting caused by morphing
        box = np.array(box) - 1  # type: ignore
        region = symbols[box[1]:box[3], box[0]:box[2]]
        count = region[region>0].size
        if count == 0:
            continue

        filled = fill_hole(region)
        f_count = filled[filled>0].size
        ratio = f_count / count

        cv2.rectangle(nn_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(nn_img, f"{ratio:.2f}", (box[2]+2, box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        if ratio > hollow_filled_ratio_th:
            hollow_box.append(box)
        else:
            solid_box.append(box)

    # Assign notes with extracted infromation
    logger.info("Instanitiating notes")
    solid_notes = gen_notes(solid_box, symbols)  # type: ignore
    hollow_notes = gen_notes(hollow_box, symbols)  # type: ignore

    logger.debug("Setting temporary note type")
    for idx in range(len(hollow_notes)):
        hollow_notes[idx].label = NoteType.HALF_OR_WHOLE

    logger.debug("Parsing whether stem is on the right")
    notes = solid_notes + hollow_notes
    parse_stem_info(notes)

    return notes


def draw_notes(notes, ori_img):
    img = ori_img.copy()
    img = np.array(img)
    for note in notes:
        x1, y1, x2, y2 = note.bbox
        x_offset = 0
        y_offset = 0
        cv2.rectangle(img, (x1+x_offset, y1+y_offset), (x2+x_offset, y2+y_offset), (0, 255, 0), 2)
        if note.label:
            cv2.putText(img, note.label.name[0], (x2+2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    return img


if __name__ == "__main__":
    f_name = "wind2"
    f_name = "last"
    #f_name = "tabi"

    #temp_data(f_name)

    # staffs, zones = st_extract()
    # layers.register_layer('staffs', np.array(staffs))
    # layers.register_layer('zones', np.array(zones))

    staff = layers.get_layer('staff_pred')
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    ori_img = layers.get_layer('original_image')

    aa = np.ones(staff.shape + (3,)) * 255
    idx = np.where(notehead+stems > 0)
    aa[idx[0], idx[1]] = 0

    notes = extract()
    rr = draw_notes(notes, aa)
