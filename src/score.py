from xml.etree import ElementTree
from collections import namedtuple
import numpy as np

DEF_EXTRA_STAFF_SPACE = 3
DEF_ROT = 1.570796
DEF_STEP_RATIO = 0.25
EPSILON = 0.1

BBox = namedtuple('BBox', ['x_min', 'x_max', 'y_min', 'y_max'])
Glyph = namedtuple('Glyph', ['name', 'box'])
Staff = namedtuple('Staff', ['glyphs', 'box'])


class Score:
    def __init__(self, fn):
        assert EPSILON > 0 and DEF_STEP_RATIO != 0

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

        # Staff line thickness
        self.staff_height = int(root[1][0].text)
        # Distance between each staff
        self.staff_space = int(root[1][1].text)
        # Thickness of a complete staff
        self.staff_thickness = self.staff_height * (DEF_EXTRA_STAFF_SPACE + 9) + \
                               self.staff_space * (DEF_EXTRA_STAFF_SPACE + 8)
        # Size of the roi that will be used for every potential glyph
        self.kernel_size = self.staff_space + 2 * self.staff_height
        # Threshold before considering there is a symbol in the current column
        self.step_threshold = 7 * self.staff_height
        # Threshold before considering there is something in the roi
        self.kernel_threshold = self.kernel_size * self.staff_height * 2.5

        cols = int(root[1][2].text)
        rows = int(root[1][3].text)
        self.shape = rows, cols
        staff_positions = [int(staff.text) for staff in root[2]]
        self.staffs = []
        for pos in staff_positions:
            box = BBox(0, cols - 1, pos, pos + self.staff_thickness)
            s = Staff([], box)
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
            self.staffs[i].glyphs.append(glyph)

    def potential_glyphs(self, staff_image):
        thickness, _ = staff_image.shape
        if thickness != self.staff_thickness:
            raise ValueError('Staff image does not belong to this score. Staff thickness does not fit.')
        col_count = np.count_nonzero(staff_image == 0, axis=0)
        connected_components = self._1d_connected_comp(col_count > self.step_threshold)
        boxes = [BBox(x_min, x_max, 0, staff_image.shape[0]) for (x_min, x_max) in connected_components]
        return boxes

    def extract_inference_data(self, img, staff):
        # TODO : Extract all the rois that could potentially hold a glyph
        return

    def extract_training_data(self, img, augment=False):
        n_samples = sum([len(staff.glyphs) for staff in self.staffs])
        if augment:
            n_samples *= 5
        X, y = [None] * n_samples, [None] * n_samples
        i = 0
        for staff in self.staffs:
            for glyph in staff.glyphs:
                label = glyph.name
                try:
                    X[i] = self._extract_roi(img, glyph.box)
                    y[i] = label
                    i += 1
                    if augment:
                        X[i:(i + 4)] = self._get_surrounding_roi(img, glyph.box)
                        y[i:(i + 4)] = [label] * 4
                except ValueError:  # Can't augment data there
                    pass
        return X

    def _get_surrounding_roi(self, img, box):
        step = self.kernel_size * DEF_STEP_RATIO
        x_min, x_max, y_min, y_max = box
        boxes, rois = [] , []
        boxes.append(BBox(x_min - step, x_max - step, y_min, y_max))
        boxes.append(BBox(x_min + step, x_max + step, y_min, y_max))
        boxes.append(BBox(x_min, x_max, y_min - step, y_max - step))
        boxes.append(BBox(x_min, x_max, y_min + step, y_max + step))
        rois = [self._extract_roi(img, b) for b in boxes]
        return rois


    def _convolve_bbox(self, img, box):
        x_min, x_max, y_min, y_max = box
        n_horizontal = (x_max - x_min) / self.kernel_size
        n_vertical = (y_max - y_min) / self.kernel_size
        x_start = x_min - self.kernel_size
        y_start = y_min - self.kernel_size

        X = np.arange(0, n_horizontal) * self.kernel_size + x_start
        Y = np.arrange(0, n_vertical) * self.kernel_size + y_start

        # TODO : Finish this with as much numpy and itertools as possible
        return img

    @staticmethod
    def _extract_roi(img, box):
        x_min, x_max, y_min, y_max = box
        if y_max > img.shape[0] or x_max > img.shape[1]:
            raise ValueError('Given image does not fit in dimensions')
        return img[y_min, y_max, x_min:x_max]

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
            end = np.argmax(arr[beg:] == False) + beg - 1
            i = end + 1
            CC.append((beg, end))
        return CC
