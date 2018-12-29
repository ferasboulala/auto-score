from xml.etree import ElementTree
from collections import namedtuple
import numpy as np

DEF_EXTRA_STAFF_SPACE = 3
DEF_ROT = 1.570796
EPSILON = 0.1

BBox = namedtuple('BBox', ['xmin', 'xmax', 'ymin', 'ymax'])
Glyph = namedtuple('Glyph', ['name', 'box'])
Staff = namedtuple('Staff', ['glyphs', 'box'])


class Score:
    def __init__(self, fn):
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

        self.staff_height = int(root[1][0].text)
        self.staff_space = int(root[1][1].text)
        self.staff_thickness = self.staff_height * (DEF_EXTRA_STAFF_SPACE + 9) + \
                               self.staff_space * (DEF_EXTRA_STAFF_SPACE + 8)
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
        xmin, xmax, ymin, ymax = staff.box
        if ymax > img.shape[0] or xmax > img.shape[1]:
            raise ValueError('Given image does not fit original image')
        return img[ymin:ymax, xmin:xmax]

    def position_glyphs(self, glyphs):
        staff_centers = [self._get_bbox_center(staff.box)[1] for staff in self.staffs]
        for glyph in glyphs:
            _, y = self._get_bbox_center(glyph.box)
            i = np.argmin(np.asarray([abs(y - c) for c in staff_centers]))
            self.staffs[i].glyphs.append(glyph)

    @staticmethod
    def _get_bbox_center(box):
        xmin, xmax, ymin, ymax = box
        return (xmin + xmax) / 2 , (ymin + ymax) / 2
