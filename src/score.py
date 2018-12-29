from xml.etree import ElementTree
from collections import namedtuple
import numpy as np

DEF_EXTRA_STAFF_SPACE = 3
DEF_ROT = 1.570796
DEF_STEP_RATIO = 0.25
EPSILON = 0.1

BBox = namedtuple('BBox', ['xmin', 'xmax', 'ymin', 'ymax'])
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
        self.step_threshold = 6 * self.staff_height
        # Threshold before considering there is something in the roi
        self.kernel_threshold = self.kernel_size * self.staff_height * 2

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

    def extract_inference_data(self, img, staff):
        staff_image = self.extract_staff_image(img, staff)
        X = []
        # TODO : Extract all the rois that could potentially hold a glyph
        return X

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
                except ValueError:
                    pass
        return X

    def _get_surrounding_roi(self, img, box):
        step = self.kernel_size * DEF_STEP_RATIO
        xmin, xmax, ymin, ymax = box
        boxes, rois = [] , []
        boxes.append(BBox(xmin - step, xmax - step, ymin, ymax))
        boxes.append(BBox(xmin + step, xmax + step, ymin, ymax))
        boxes.append(BBox(xmin, xmax, ymin - step, ymax - step))
        boxes.append(BBox(xmin, xmax, ymin + step, ymax + step))
        rois = [self._extract_roi(img, b) for b in boxes]
        return rois

    @staticmethod
    def _extract_roi(img, box):
        xmin, xmax, ymin, ymax = box
        if ymax > img.shape[0] or xmax > img.shape[1]:
            raise ValueError('Given image does not fit in dimensions')
        return img[ymin:ymax, xmin:xmax]

    @staticmethod
    def _get_bbox_center(box):
        xmin, xmax, ymin, ymax = box
        return (xmin + xmax) / 2 , (ymin + ymax) / 2
