from typing import Tuple, List, Any, Dict
import random
import math

import cv2
from cv2.typing import RotatedRect
import scipy.ndimage
import numpy as np
from numpy import ndarray

from oemer import layers
from oemer.utils import get_unit_size
from oemer.logger import get_logger
from oemer.bbox import BBox, get_center, get_rotated_bbox, to_rgb_img, draw_bounding_boxes
from oemer.notehead_extraction import NoteType
from oemer.morph import morph_open, morph_close

# Globals
dot_img: ndarray
ratio_img: ndarray
beam_img: ndarray
ratio_map: ndarray

logger = get_logger(__name__)


def scan_dot(
        symbols: ndarray, 
        note_id_map: ndarray, 
        bbox: BBox, 
        unit_size: float, 
        min_count: int, 
        max_count: int
) -> bool:
    right_bound = bbox[2] + 1
    start_y = bbox[1] - round(unit_size / 2)
    while True:
        # Find the right most bound for scan the dot.
        # Should have width less than unit_size, and can't
        # touch the nearby note.
        try:
            cur_scan_line = note_id_map[int(start_y):int(bbox[3]), int(right_bound)]
        except IndexError as e:
            print(e)
            break
        
        ids = set(np.unique(cur_scan_line))
        if -1 in ids:
            ids.remove(-1)
        if len(ids) > 0:
            break
        right_bound += 1
        if right_bound >= bbox[2] + unit_size:
            break

    left_bound = bbox[2] + round(unit_size*0.4)
    dot_region = symbols[int(start_y):int(bbox[3]), int(left_bound):int(right_bound)]
    pixels = np.sum(dot_region)
    if min_count <= pixels <= max_count:
        color = (255, random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(dot_img, (int(left_bound), int(start_y)), (int(right_bound), int(bbox[3])), color, 1)
        msg = f"{min_count:.2f}/{pixels:.2f}/{max_count:.2f}"
        cv2.putText(dot_img, msg, (int(bbox[0]), int(bbox[3])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        return True

    # color = (255, random.randint(0, 255), random.randint(0, 255))
    # cv2.rectangle(temp, (bbox[2]+1, start_y), (right_bound, bbox[3]), color, 1)
    # if pixels > 0:
    #     msg = f"{min_count:.2f}/{pixels:.2f}/{max_count:.2f}"
    #     cv2.putText(temp, msg, (bbox[0], bbox[3]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    return False


def parse_dot(min_area_ratio: float = 0.08, max_area_ratio: float = 0.2) -> None:
    # Fetch parameters
    groups = layers.get_layer('note_groups')
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    clefs_sfns = layers.get_layer('clefs_keys_pred')
    notes = layers.get_layer('notes')
    note_id_map = layers.get_layer('note_id')

    # Post-process necessary variables
    no_stem = np.where(symbols-stems-clefs_sfns>0, 1, 0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    no_stem = cv2.dilate(cv2.erode(no_stem.astype(np.uint8), ker), ker)
    global dot_img
    dot_img = to_rgb_img(no_stem)

    # Find the dot besides each note
    for group in groups:
        nids = group.note_ids
        gbox = group.bbox
        unit_size = get_unit_size(*get_center(gbox))
        nbox = np.array([notes[nid].bbox for nid in nids])
        nbox = (np.min(nbox[:, 0]), np.min(nbox[:, 1]), np.max(nbox[:, 2]), np.max(nbox[:, 3]))  # type: ignore
        min_count = round(unit_size**2 * min_area_ratio)
        max_count = round(unit_size**2 * max_area_ratio)

        dots = []
        for nid in nids:
            bbox = notes[nid].bbox
            bbox = (bbox[0], bbox[1], max(bbox[2], gbox[2]), bbox[3])
            has_dot = scan_dot(no_stem, note_id_map, bbox, unit_size, min_count, max_count)
            dots.append(has_dot)
            notes[nid].has_dot = has_dot

        all_same = not (all(dots) ^ any(dots))  # Notes all have dot or not
        if group.stem_up is not None:
            if not all_same:
                # Should have the same rhythm type
                true_count = len([dot for dot in dots if dot])
                false_count = len(dots) - true_count
                to_dot = (true_count >= false_count)
                for nid in nids:
                    notes[nid].has_dot = to_dot


def polish_symbols(
        staff_pred: ndarray, 
        symbols: ndarray, 
        stems: ndarray, 
        clefs_sfns: ndarray, 
        group_map: ndarray
) -> ndarray:
    st_width = 5
    beams_in_staff = morph_open(staff_pred, (st_width, 1))

    gp_map = np.where(group_map>-1, 1, 0)
    mix = symbols + beams_in_staff - gp_map - stems - clefs_sfns
    mix = np.where(mix > 0, 1, 0)
    mix = morph_open(mix, (2, 3))  # Remove possible false connections between beams and slurs
    ext_stems = morph_close(stems, (5, 1))
    beams = mix + ext_stems + gp_map
    beams[beams>1] = 1
    return beams


def parse_beams(
        min_area_ratio: float = 0.07, 
        min_tp_ratio: float = 0.4, 
        min_width_ratio: float = 0.2
) -> Tuple[ndarray, List[RotatedRect], ndarray]:
    # Fetch parameters
    symbols = layers.get_layer('symbols_pred')
    staff_pred = layers.get_layer('staff_pred')
    stems = layers.get_layer('stems_rests_pred')
    group_map = layers.get_layer('group_map')
    clefs_sfns = layers.get_layer('clefs_keys_pred')

    beams = polish_symbols(staff_pred, symbols, stems, clefs_sfns, group_map)
    beams = beams - np.where(group_map>-1, 1, 0) - stems
    beams[beams<0] = 0

    rboxes = get_rotated_bbox(beams)
    poly_map = np.ones(symbols.shape+(3,), dtype=np.uint8) * 255
    idx = np.where(beams>0)
    poly_map[idx[0], idx[1]] = 0
    invalid_map = np.zeros_like(poly_map)  # Used to eliminate elements in 'symbols' of later stage.

    global ratio_map
    ratio_map = np.copy(poly_map)

    null_color = (255, 255, 255)
    valid_box: List[RotatedRect] = []
    valid_idxs = []
    idx_map = np.zeros_like(poly_map) - 1
    for box_idx, rbox in enumerate(rboxes):  # type: ignore
        # Used to find indexes of contour areas later. Must be check before
        # any 'continue' statement.
        box_idx %= 255  # type: ignore
        if box_idx == 0:
            idx_map = np.zeros_like(poly_map) - 1

        # Get the contour of the rotated box
        cnt = cv2.boxPoints(rbox)
        if any(cc < 0 for cc in cnt.reshape(-1, 1).squeeze()):
            # Check there is no negative points that could further
            # cause overflow while converting to unsigned int.
            continue
        cnt = cnt.astype(np.uint64)
        centers = np.sum(cnt, axis=0) / 4
        unit_size = get_unit_size(round(centers[0]), round(centers[1]))

        # Check the area size
        cv2.drawContours(ratio_map, [cnt], 0, (255, 0, 0), 2)
        area = cv2.contourArea(cnt)
        min_area = unit_size**2 * min_area_ratio
        if area < min_area:
            cv2.fillPoly(poly_map, [cnt], color=null_color)
            cv2.fillPoly(invalid_map, [cnt], color=null_color)
            continue

        # Tricky way to get the index of the contour area
        cv2.fillPoly(idx_map, [cnt], color=(box_idx, 0, 0))
        yi, xi = np.where(idx_map[..., 0] == box_idx)
        pts = beams[yi, xi]
        meta_idx = np.where(pts>0)[0]
        ryi = yi[meta_idx]
        rxi = xi[meta_idx]

        # Check the width of the rotated box
        r_width = min(rbox[1])
        if r_width < unit_size * min_width_ratio:
            poly_map[ryi, rxi] = np.array(null_color)
            invalid_map[ryi, rxi] = np.array(null_color)
            ratio_map[ryi, rxi] = np.array((255, 235, 15))
            continue

        # Check true area ratio
        ratio = len(meta_idx) / (len(yi) + 1e-8)
        if ratio < min_tp_ratio:
            poly_map[ryi, rxi] = np.array(null_color)
            invalid_map[ryi, rxi] = np.array(null_color)
            ratio_map[ryi, rxi] = np.array((0, 150, 255))
            cv2.putText(ratio_map, f"{ratio:.2f}", (cnt[1][0], cnt[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            continue

        valid_idxs.append((ryi, rxi))
        valid_box.append(rbox)

    for ryi, rxi in valid_idxs:
        poly_map[ryi, rxi] = 255 - np.array(null_color)
        ratio_map[ryi, rxi] = 255 - np.array(null_color)

    poly_map = np.where(np.sum(poly_map, axis=-1)<1, 1, 0)
    invalid_map = invalid_map[..., 0] / 255
    return poly_map, valid_box, invalid_map


def parse_beam_overlap_regions(poly_map: ndarray, invalid_map: ndarray) -> Tuple[ndarray, Dict[int, Dict[str, Any]]]:
    # Fetch parameters
    symbols = layers.get_layer('symbols_pred')
    group_map = layers.get_layer('group_map')
    barlines = layers.get_layer('barlines')

    mix = poly_map + symbols #- invalid_map
    mix[mix<0] = 0
    ker = np.ones((3, 3), dtype=np.uint8)
    mix = cv2.dilate(cv2.erode(mix.astype(np.uint8), ker), ker)
    poly_map = cv2.dilate(cv2.erode(poly_map.astype(np.uint8), ker), ker)

    # Remove barlines
    for bl in barlines:
        box = bl.bbox
        mix[box[1]:box[3], box[0]:box[2]] = 0
        poly_map[box[1]:box[3], box[0]:box[2]] = 0

    mix[mix>1] = 1
    reg_map, feat_num = scipy.ndimage.label(poly_map)
    sym_map, _ = scipy.ndimage.label(mix)

    out_map = np.zeros_like(reg_map)
    map_info: Dict = {}
    for idx in range(1, feat_num+1):
        mask = (reg_map == idx)
        sym_labels = set(np.unique(sym_map[mask]))
        if 0 in sym_labels:
            sym_labels.remove(0)

        yi, xi = [], []
        for label in sym_labels:
            yy, xx = np.where(sym_map==label)
            yi.extend(list(yy))
            xi.extend(list(xx))

        g_overlap = group_map[np.array(yi).astype(int), np.array(xi).astype(int)]
        gids = set(np.unique(g_overlap))
        if -1 in gids:
            gids.remove(-1)
        if len(gids) == 0:
            # Discard the region that doesn't overlap with any note group.
            continue

        out_map[mask] = 1
        box = (np.min(xi), np.min(yi), np.max(xi), np.max(yi))
        for sym_label in sym_labels:
            if sym_label in map_info:
                bb = map_info[sym_label]['bbox']
                gids.update(map_info[sym_label]['gids'])
                box = (min(bb[0], box[0]), min(bb[1], box[1]), max(bb[2], box[2]), max(bb[3], box[3]))
            map_info[sym_label] = {'bbox': box, 'gids': gids}

    ker = np.ones((3, 3), dtype=np.uint8)
    out_map = cv2.erode(cv2.dilate(out_map.astype(np.uint8), ker), ker)  # Make it more smooth
    return out_map, map_info


def refine_map_info(map_info: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    # Fetch parameters
    groups = layers.get_layer('note_groups')
    group_map = layers.get_layer('group_map')

    new_map_info: Dict = {}
    rev_map = {}
    for reg, info in map_info.items():
        cur_gids = info['gids']
        bbox = info['bbox']
        new_map_info[reg] = {'bbox': None, 'gids': set()}

        bbox = np.array([groups[gid].bbox for gid in cur_gids] + [bbox])
        bbox = (np.min(bbox[:, 0]), np.min(bbox[:, 1]), np.max(bbox[:, 2]), np.max(bbox[:, 3]))
        new_map_info[reg]['bbox'] = bbox
        region = group_map[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        gids = set(np.unique(region))
        if -1 in gids:
            gids.remove(-1)

        for gid in gids:
            if gid not in rev_map:
                new_map_info[reg]['gids'].add(gid)
                rev_map[gid] = reg
            else:
                ori_reg = rev_map[gid]
                if ori_reg == reg:
                    continue

                ori_bbox = new_map_info[ori_reg]['bbox']
                ori_gids = new_map_info[ori_reg]['gids']
                new_box = (
                    min(ori_bbox[0], bbox[0]),
                    min(ori_bbox[1], bbox[1]),
                    max(ori_bbox[2], bbox[2]),
                    max(ori_bbox[3], bbox[3])
                )
                ori_gids.add(gid)
                new_map_info[reg]['gids'].update(ori_gids)
                new_map_info[reg]['bbox'] = new_box
                rev_map[gid] = reg

                del new_map_info[ori_reg]
                for ogid in ori_gids:
                    rev_map[ogid] = reg

    return new_map_info


def get_stem_x(gbox: BBox, nboxes: List[ndarray], unit_size: float, is_right: bool = True) -> int:
    all_same_side = all(abs(nb[2]-gbox[2])<unit_size/3 for nb in nboxes)
    stem_at_center = not all_same_side
    if stem_at_center:
        return round((gbox[0] + gbox[2]) / 2)
    elif is_right:
        return gbox[2]
    else:
        return gbox[0]


def scan_beam_flag(
    poly_map: ndarray,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    threshold: float = 0.1,
    min_width_ratio: float = 0.25,
    max_width_ratio: float = 0.9
) -> int:

    start_x = int(start_x)
    start_y = int(start_y)
    end_x = int(end_x)
    end_y = int(end_y)

    cv2.line(beam_img, (start_x, start_y), (end_x, start_y), (42, 110, 200), 2, cv2.LINE_8)
    cv2.line(beam_img, (start_x, end_y), (end_x, end_y), (42, 110, 200), 2, cv2.LINE_8)

    counter = [0 for _ in range(end_x-start_x)]

    if end_y < start_y:
        start_y, end_y = end_y, start_y

    unit_size = get_unit_size(start_x, start_y)
    min_width = unit_size * min_width_ratio
    max_width = unit_size * max_width_ratio
    for idx, x in enumerate(range(start_x, end_x)):
        cur_y = start_y
        last_val = int(poly_map[cur_y, x])

        # Start scanning
        while cur_y < end_y:
            hit = False
            width = 0
            while cur_y < end_y:
                cur_val = int(poly_map[cur_y, x])
                if last_val ^ cur_val:
                    hit = last_val > cur_val
                    last_val = cur_val
                    cur_y += 1
                    break
                cur_y += 1
                width += 1
            if hit and width >= min_width:
                beam_count = math.ceil(width / max_width)
                counter[idx] += beam_count
        if last_val == 1:
            # Not yet hit a change but loop ended.
            beam_count = math.ceil(width / max_width) if hit else 1
            counter[idx] += beam_count

    # Summarize counts
    stat = {}
    for c in counter:
        if c not in stat:
            stat[c] = 0
        stat[c] += 1
    stat = sorted(stat.items(), key=lambda s: s[0], reverse=True)  # type: ignore

    # At least there are such amount agreed with that there
    # are this number of beams/flags.
    accum = 0
    min_num = len(counter) * threshold
    for c, num in stat:  # type: ignore
        accum += num  # type: ignore
        if accum > min_num:
            return c
    return 0


def parse_inner_groups(poly_map, group, set_box, note_type_map, half_scan_width, threshold=0.1):
    # Fetch parameters
    notes = layers.get_layer('notes')

    nts = np.copy([notes[nid] for nid in group.note_ids])  # Copy to avoid messing the order.
    nts = sorted(nts, reverse=True)  # Will sort by staffline position

    def get_label(nbox, stem_up):
        cen_x = nbox[2] if stem_up else nbox[0]
        start_y = nbox[1] if stem_up else nbox[3]
        end_y = set_box[1] if stem_up else set_box[3]
        count = scan_beam_flag(
            poly_map,
            start_x=max(set_box[0], cen_x-half_scan_width),
            start_y=start_y,
            end_x=min(set_box[2], cen_x+half_scan_width),
            end_y=end_y,
            threshold=threshold
        )        
        if count >= len(note_type_map):
            return note_type_map[len(note_type_map) - 1]
        return note_type_map[count]

    if len(nts) == 2:
        # One note has stem up, and the other down.
        notes[nts[0].id].force_set_label(get_label(nts[0].bbox, stem_up=True))
        notes[nts[1].id].force_set_label(get_label(nts[1].bbox, stem_up=False))
        notes[nts[0].id].stem_up = True
        notes[nts[1].id].stem_up = False
        group.top_note_ids.append(nts[0].id)
        group.bottom_note_ids.append(nts[1].id)
    elif group.all_same_type:
        # Either all the notes are solid or hollow.
        # Always assume there is only one note at top, and others are bottom.
        notes[nts[0].id].label = get_label(nts[0].bbox, stem_up=True)
        notes[nts[0].id].stem_up = True
        group.top_note_ids.append(nts[0].id)
        bt_label = get_label(nts[-1].bbox, stem_up=False)
        for nn in nts[1:]:
            notes[nn.id].label = bt_label
            notes[nn.id].stem_up = False
            group.bottom_note_ids.append(nn.id)
    else:
        # The most complicated situation, which contains both solid and hollow notes.
        # Need to find the split position that separates solid and hollow part.
        idx = 0
        while idx < len(nts):
            if nts[0].label != nts[idx].label:
                break
            idx += 1

        if nts[0].label == NoteType.HALF_OR_WHOLE:
            # The top group is half notes.
            for nn in nts[:idx]:
                #assert nn.label == NoteType.HALF_OR_WHOLE
                notes[nn.id].force_set_label(NoteType.HALF)
                notes[nn.id].stem_up = True
                group.top_note_ids.append(nn.id)
            bt_label = get_label(nts[-1].bbox, stem_up=False)
            for nn in nts[idx:]:
                notes[nn.id].label = bt_label
                notes[nn.id].stem_up = False
                group.bottom_note_ids.append(nn.id)
        else:
            # The bottom group is half notes.
            for nn in nts[idx:]:
                #assert nn.label == NoteType.HALF_OR_WHOLE, nn
                notes[nn.id].force_set_label(NoteType.HALF)
                notes[nn.id].stem_up = False
                group.bottom_note_ids.append(nn.id)
            top_label = get_label(nts[-1].bbox, stem_up=True)
            for nn in nts[:idx]:
                notes[nn.id].label = top_label
                notes[nn.id].stem_up = True
                group.top_note_ids.append(nn.id)


def parse_rhythm(beam_map: ndarray, map_info: Dict[int, Dict[str, Any]], agree_th: float = 0.15) -> ndarray:
    # Fetch parameters
    groups = layers.get_layer('note_groups')
    notes = layers.get_layer('notes') 
    notehead = layers.get_layer('notehead_pred')

    # Collect neccessary information
    rev_map_info = {}
    for reg, info in map_info.items():
        gids = info['gids']
        box = info['bbox']
        for gid in gids:
            rev_map_info[gid] = {'reg': reg, 'bbox': box}

    # Define beam count to note type mapping
    note_type_map: Dict[int, NoteType] = {
        0: NoteType.QUARTER,
        1: NoteType.EIGHTH,
        2: NoteType.SIXTEENTH,
        3: NoteType.THIRTY_SECOND,
        4: NoteType.SIXTEENTH,
        #5: None,
        #6: None
    }

    global beam_img
    beam_img = to_rgb_img(np.where(beam_map+notehead>0, 1, 0))
    # bboxes = [v['bbox'] for v in map_info.values()]
    # beam_img = draw_bounding_boxes(bboxes, beam_img)

    # Start parsing the rhythm
    bin_beam_map = np.where(beam_map>0, 1, 0)
    for gid in range(len(groups)):
        group = groups[gid]
        gbox = group.bbox
        reg_box = rev_map_info[gid]['bbox'] if gid in rev_map_info else gbox
        unit_size = get_unit_size(*get_center(gbox))
        half_scan_width = round(unit_size / 2)

        # Check stem status
        if group.stem_up is None:
            if not group.has_stem:
                # Could only be whole note
                for nid in group.note_ids:
                    if notes[nid].label != NoteType.HALF_OR_WHOLE:
                        notes[nid].invalid = True
                        continue
                    notes[nid].force_set_label(NoteType.WHOLE)
            else:
                parse_inner_groups(
                    poly_map=bin_beam_map,
                    group=group,
                    set_box=reg_box,
                    note_type_map=note_type_map,
                    half_scan_width=half_scan_width,
                    threshold=agree_th
                )
            continue

        # Conclude note types of notes in this group
        labels = [notes[nid].label for nid in group.note_ids]
        count = {k: 0 for k in set(labels)}
        for l in labels:
            count[l] += 1
        count = sorted(count.items(), key=lambda c: c[1], reverse=True)  # type: ignore
        label = count[0][0]  # type: ignore
        if label == NoteType.HALF_OR_WHOLE:
            # This group contians only half notes
            for nid in group.note_ids:
                notes[nid].force_set_label(NoteType.HALF)
            continue

        if gid not in rev_map_info:
            # Has no beams or flags attached to this group, thus
            # could only be quarter notes
            for nid in group.note_ids:
                #assert notes[nid].label is None, notes[nid]
                notes[nid].label = NoteType.QUARTER
            continue

        gbox = group.bbox
        gbox = (gbox[0], min(gbox[1], reg_box[1]), gbox[2], max(gbox[3], reg_box[3]))  # Adjust only y-axis
        nbox = [notes[nid].bbox for nid in group.note_ids]
        unit_size = get_unit_size(*get_center(gbox))
        if group.stem_up:
            cen_x = get_stem_x(gbox, nbox, unit_size)
            start_y = min(nb[1] for nb in nbox)
            end_y = gbox[1]
        else:
            cen_x = get_stem_x(gbox, nbox, unit_size, is_right=False)
            start_y = max(nb[3] for nb in nbox)
            end_y = gbox[3]

        # Calculate how many beams/flags are there.
        beam_flag_count = scan_beam_flag(  # type: ignore
            bin_beam_map,
            max(reg_box[0], cen_x-half_scan_width),
            start_y,
            min(reg_box[2], cen_x+half_scan_width),
            end_y,
            threshold=agree_th
        )

        #cv2.rectangle(beam_img, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (255, 0, 255), 1)
        cv2.putText(beam_img, str(beam_flag_count), (int(cen_x), int(gbox[3])+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Assign note label
        for nid in group.note_ids:
            if notes[nid].label is None:
                if beam_flag_count in note_type_map:
                    notes[nid].label = note_type_map[beam_flag_count]
                else:
                    notes[nid].invalid = True

    return beam_img


def extract(
    dot_min_area_ratio: float = 0.08,
    dot_max_area_ratio: float = 0.2,
    beam_min_area_ratio: float = 0.07,
    agree_th: float = 0.15
) -> Tuple[ndarray, List[RotatedRect]]:

    logger.debug("Parsing dot")
    parse_dot(max_area_ratio=dot_max_area_ratio, min_area_ratio=dot_min_area_ratio)

    logger.debug("Parsing beams and flags")
    poly_map, valid_box, invalid_map = parse_beams(min_area_ratio=beam_min_area_ratio)

    logger.debug("Parsing beam regions")
    out_map, map_info = parse_beam_overlap_regions(poly_map, invalid_map)

    logger.debug("Refining the map info")
    map_info = refine_map_info(map_info)

    logger.debug("Parsing notes rhythm")
    beam_img = parse_rhythm(out_map, map_info, agree_th=agree_th)
    return beam_img, valid_box


def draw_notes(notes, ori_img):
    img = ori_img.copy()
    img = np.array(img)
    for note in notes:
        x1, y1, x2, y2 = note.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if note.label is not None:
            cv2.putText(img, note.label.name[0], (x2+2, y2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)
        else:
            cv2.putText(img, "invalid", (x2+2, y2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2)
        if note.has_dot:
            cv2.putText(img, "DOT", (x2+17, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1)
        # cv2.putText(img, str(note.stem_up), (x2+2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    return img


if __name__ == "__main__":
    ori_img = layers.get_layer('original_image')
    notes = layers.get_layer('notes')
    groups = layers.get_layer('note_groups')
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    clefs_sfns = layers.get_layer('clefs_keys_pred')
    note_id_map = layers.get_layer('note_id')
    staff = layers.get_layer('staff_pred')
    notehead = layers.get_layer('notehead_pred')
    group_map = layers.get_layer('group_map')

    logger.debug("Parsing dot")
    parse_dot()

    logger.debug("Parsing beams and flags")
    poly_map, valid_box, invalid_map = parse_beams()

    logger.debug("Parsing beam regions")
    out_map, map_info = parse_beam_overlap_regions(poly_map, invalid_map)

    logger.debug("Refining the map info")
    map_info = refine_map_info(map_info)

    logger.debug("Parsing notes rhythm")
    beam_img = parse_rhythm(out_map, map_info)

    out = draw_notes(notes, ori_img)

    bboxes = [info['bbox'] for info in map_info.values()]
    #beam_img = draw_bounding_boxes(bboxes, beam_img)
