# TODO : Add a function that saves all training data
# TODO : Repeat all the processes for the muscima dataset
# TODO : Add comments a docstrings

from xml.etree import ElementTree
from collections import namedtuple
from os.path import join, splitext

import numpy as np

DEF_EXTRA_STAFF_SPACE = 4
DEF_ROT = 1.570796
DEF_STEP_RATIO = 0.25
EPSILON = 0.1

BBox = namedtuple('BBox', ['x_min', 'x_max', 'y_min', 'y_max'])
Glyph = namedtuple('Glyph', ['name', 'box'])
Staff = namedtuple('Staff', ['glyphs', 'box', 'start'])


class Score:
    def __init__(self, fn, no_staff_lines=True):
        tree = ElementTree.parse(fn)
        root = tree.getroot()
        if root.tag != 'stav':
            raise ValueError('XML file not in the expected format')

        rotation = float(root[1][4].text)
        if abs(rotation - DEF_ROT) > EPSILON:
            raise ValueError('Staff lines must be all straight')

        model_gradient = np.asarray([float(gradient) for gradient in str.split(root[1][5].text)])
        if np.count_nonzero(model_gradient):
            raise ValueError('Staff model must be rectified')

        self.staff_length = len(model_gradient)
        self.filename = root[0].text

        # Staff line thickness
        self.staff_height = int(root[1][0].text)
        # Distance between each staff
        self.staff_space = int(root[1][1].text)
        # Thickness of a complete staff
        self.staff_extra = (self.staff_height + self.staff_space) * DEF_EXTRA_STAFF_SPACE
        self.staff_thickness = self.staff_height * 5 + self.staff_space * 4
        # Size of the roi that will be used for every potential glyph
        self.kernel_size = self.staff_space + 2 * self.staff_height
        # Threshold before considering there is a symbol in the current column
        self.step_threshold = 7 * self.staff_height
        if no_staff_lines:
            self.step_threshold -= 5 * self.staff_height
        # Threshold before considering there is something in the roi
        self.kernel_threshold = self.kernel_size * self.staff_height * 2
        if no_staff_lines:
            self.kernel_threshold -= self.kernel_size * self.staff_height
        # Convolution
        self.step = self.kernel_size // 4

        cols = int(root[1][2].text)
        rows = int(root[1][3].text)
        self.shape = rows, cols
        self.staff_positions = [int(staff.text) + rows for staff in root[2]]
        self.staffs = []
        for pos in self.staff_positions:
            box = BBox(0, cols - 1, pos - self.staff_extra // 2,
            pos + self.staff_thickness + self.staff_extra // 2)
            s = Staff([], box, pos)
            self.staffs.append(s)

    def __getitem__(self, item):
        return self.staffs[item]

    def __len__(self):
        return len(self.staffs)

    def extract_staff_image(self, img, staff):
        return self._extract_roi(img, staff.box)

    def position_glyphs(self, glyphs):
        staff_centers = [self._get_bbox_center(staff.box)[1] for staff in self.staffs]
        for glyph in glyphs:
            _, y = self._get_bbox_center(glyph.box)
            i = np.argmin(np.asarray([abs(y - c) for c in staff_centers]))
            x_min, x_max, y_min, y_max = glyph.box
            label = glyph.name
            y_min -= self.staffs[i].box.y_min
            y_max -= self.staffs[i].box.y_min
            positioned_glyph = Glyph(label, BBox(x_min, x_max, y_min, y_max))
            self.staffs[i].glyphs.append(positioned_glyph)

    def potential_regions(self, staff_image, merge=False, thin_filter=False):
        thickness, _ = staff_image.shape
        if thickness != self.staff_thickness + self.staff_extra:
            raise ValueError('Staff image does not belong to this score. Staff thickness does not fit.')
        col_count = np.count_nonzero(staff_image == 0, axis=0)
        connected_components = self._1d_connected_comp(col_count > self.step_threshold)
        if merge:
            connected_components = self._1d_merge_cc(connected_components, self.kernel_size / 2)
        if thin_filter:
            connected_components = self.filter_cc(connected_components, self.kernel_size)
        boxes = [BBox(x_min, x_max, 0, staff_image.shape[0]) for (x_min, x_max) in connected_components]
        return boxes

    def potential_glyphs(self, staff_image, regions):
        potential_glyphs = []
        for region in regions:
            glyphs = self.convolve_box(region)
            for x_min, x_max, y_min, y_max in glyphs:
                area = staff_image[y_min:y_max, x_min:x_max]
                if np.count_nonzero(area == 0) > self.kernel_threshold:
                    potential_glyphs.append(BBox(x_min, x_max, y_min, y_max))
        return potential_glyphs

    def extract_training_data(self, staff_image, staff, relevant_glyphs):
        poll_map = np.ones_like(staff_image)  # How many
        label_map = np.zeros_like(staff_image)  # What id

        if 'None' in relevant_glyphs:
            raise ValueError('None type is reserved for the extracting algorithm')

        glyph_to_id = {name: (i+1) for i, name in enumerate(relevant_glyphs)}
        id_to_glyph = {(i+1): name for i, name in enumerate(relevant_glyphs)}

        for glyph in staff.glyphs:
            x_min, x_max, y_min, y_max = glyph.box
            if glyph.name not in glyph_to_id:
                continue

            gt_image = staff_image[y_min:y_max, x_min:x_max]
            mask = gt_image == 0
            gt_count = np.sum(mask)

            region_poll = poll_map[y_min:y_max, x_min:x_max]
            region_poll[mask] = gt_count
            region_label = label_map[y_min:y_max, x_min:x_max]
            region_label[mask] = glyph_to_id[glyph.name]

        potential_regions = self.potential_regions(staff_image, True, True)
        potential_glyphs = self.potential_glyphs(staff_image, potential_regions)
        X, y = [], []
        for x_min, x_max, y_min, y_max in potential_glyphs:
            label = 'None'
            region_poll = poll_map[y_min:y_max, x_min:x_max]
            region_label = label_map[y_min:y_max, x_min:x_max]
            not_none = region_label[region_label != 0]

            if not_none.size == 0:
                X.append(BBox(x_min, x_max, y_min, y_max))
                y.append(label)
                continue

            counts = np.bincount(not_none)
            count = np.max(counts)
            glyph_id = np.argmax(counts)
            gt_count = np.where(region_label == glyph_id)
            i, j = gt_count[0][0], gt_count[1][0]

            if count > 0.5 * region_poll[i, j]:
                label = id_to_glyph[glyph_id]

            X.append(BBox(x_min, x_max, y_min, y_max))
            y.append(label)

        return X, y


    def convolve_box(self, box):
        x_min, x_max, y_min, y_max = box
        n_horizontal = (x_max - x_min + self.step * 2) // self.step
        n_vertical = (y_max - y_min + self.step * 2) // self.step
        x_start = x_min - self.step
        y_start = y_min - self.step

        X = np.arange(0, n_horizontal) * self.step + x_start
        Y = np.arange(0, n_vertical) * self.step + y_start

        return [BBox(x, x + self.kernel_size, y, y + self.kernel_size)
                for x in X for y in Y if x >= 0 and y >= 0]

    @staticmethod
    def filter_cc(cc, threshold):
        return [(start, finish) for start, finish in cc if finish - start >= threshold]

    @staticmethod
    def _extract_roi(img, box):
        x_min, x_max, y_min, y_max = box
        rows, cols, *_ = img.shape

        if y_max > rows:
            y_max = rows
        elif y_min < 0:
            y_min = 0
        if x_max > cols:
            y_max = cols
        elif x_min < 0:
            x_min = 0

        return img[y_min:y_max, x_min:x_max]

    @staticmethod
    def _get_bbox_center(box):
        x_min, x_max, y_min, y_max = box
        return (x_min + x_max) / 2 , (y_min + y_max) / 2

    @staticmethod
    def _1d_connected_comp(arr):
        i = 0
        CC = []
        beg, end = 0, 0
        while beg <= end:
            beg = np.argmax(arr[i:]) + i
            end = np.argmax(~arr[beg:]) + beg - 1
            i = end + 1
            CC.append((beg, end))
        return CC

    @staticmethod
    def _1d_merge_cc(cc, threshold):
        if len(cc) == 1:
            return cc
        merged = []
        i = 0
        while i < len(cc) - 1:
            start, finish = cc[i]
            while cc[i + 1][0] - finish <= threshold:
                finish = cc[i + 1][1]
                if i == len(cc) - 2:
                    break
                i += 1
            merged.append((start, finish))
            i += 1
        return merged


def deepscores_gt(score_filename, dataset_filename):
    tree = ElementTree.parse(join(dataset_filename, splitext(score_filename)[0] + '.xml'))
    root = tree.getroot()
    if root.tag != 'annotation':
        raise ValueError('This is not a valid deepscores ground truth description file')

    width = height = 0
    glyphs = []
    for node in root:
        if node.tag == 'size':
            width = int(node[0].text)
            height = int(node[1].text)
            continue

        elif node.tag != 'object':
            continue

        name = node[0].text

        x_min, x_max = int(width * float(node[1][0].text)), int(width * float(node[1][1].text))
        y_min, y_max = int(height * float(node[1][2].text)), int(height * float(node[1][3].text))
        bbox = BBox(x_min, x_max, y_min, y_max)
        glyph = Glyph(name, bbox)
        glyphs.append(glyph)

    return glyphs
