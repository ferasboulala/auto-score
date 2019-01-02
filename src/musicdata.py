"""
This module is the initial attempt at generating ground truth for OMR.
The score module is the preferred method.
"""

import cv2 as cv
import xml.etree.ElementTree as ET
from collections import namedtuple
from os.path import join, splitext, isfile, expanduser
from os import listdir
from subprocess import call

PREFIX_FN = expanduser("~") + '/Documents/auto-score/datasets/'
ARTIFICIAL_FN = PREFIX_FN + 'Artificial/'
HANDWRITTEN_FN = PREFIX_FN + 'Handwritten/'
ARTIFICIAL_FN_XML = ARTIFICIAL_FN + 'xml/'
HANDWRITTEN_FN_XML = HANDWRITTEN_FN + 'xml/'
ARTIFICIAL_FN_DATA = ARTIFICIAL_FN + 'data/'
HANDWRITTEN_FN_DATA = HANDWRITTEN_FN + 'data/'

# Musical symbol called a glyph
Glyph = namedtuple('Glyph', 'name bbox')

# Bounding box of an entity in an image
BBox = namedtuple('Bbox', 'xmin xmax ymin ymax')

# Portion of the kernel that is exclusive to the previous one
STRIDE_RATIO = 0.5
# Amount of staff space added over the staff lines
BOUNDARY_EXTRA = 4
# Minimum amount of samples for training
POLL_THRESH = 50
# Useful constant
EPSILON = 1e-7
# Area overlap
AREA_OVERLAP_MIN = 0.6

RED = 255, 0, 0
BLUE = 0, 0, 255
GREEN = 0, 255, 0

RELEVANT_GLYPHS_MUSCIMA = {'g-clef', 'sharp', 'notehead-full', 'flat', 'natural', 'notehead-empty', 'None', 'Other'}
RELEVANT_GLYPHS_DEEPSCORES = {'noteheadBlack', 'noteheadHalf', 'gClef', 'keySharp', 'accidentalNatural',
                              'noteheadWhole','accidentalSharp', 'None', 'Other'}


def _setup_dir(RELEVANT_GLYPHS, dir_fn):
    call(['mkdir', dir_fn])
    for name in RELEVANT_GLYPHS:
        call(['mkdir', dir_fn + name])


def _get_music_files_from_dir(dir_fn):
    filenames = [f for f in listdir(dir_fn) if isfile(join(dir_fn, f))]
    return [get_music_file_from_xml(join(dir_fn, f)) for f in filenames]


def _save_samples_to_disk(X, y, glyph_count, dir_fn):
    for img, label in zip(X, y):
        glyph_count[label] += 1
        fn = dir_fn + label + '/' + str(glyph_count[label]) + '.jpg'
        cv.imwrite(fn, img)


def get_deepscores_data(images_dir, gt_dir):
    _setup_dir(RELEVANT_GLYPHS_DEEPSCORES, ARTIFICIAL_FN_DATA)
    music_files = _get_music_files_from_dir(ARTIFICIAL_FN_XML)
    glyphs_per_score = []
    content = dict()
    for music_file in music_files:
        glyph_list = deepscores_score_ground_truth(music_file.filename, gt_dir)
        glyphs_per_score.append(glyph_list)

    for glyphs in glyphs_per_score:
        for glyph in glyphs:
            if glyph.name not in content:
                content[glyph.name] = 1
            else:
                content[glyph.name] += 1

    glyph_count = dict((name, 0) for name in RELEVANT_GLYPHS_DEEPSCORES)
    total_count = 0
    for i, music_file in enumerate(music_files):
        music_file.position_glyphs(glyphs_per_score[i], content)
        img = cv.imread(join(images_dir, music_file.filename), cv.CV_8UC1)
        X, y = music_file.extract_training_data(img, content, RELEVANT_GLYPHS_DEEPSCORES)
        _save_samples_to_disk(X, y, glyph_count, ARTIFICIAL_FN_DATA)
        total_count += 1
        print('{0} processed'.format(total_count))


def get_muscima_data(images_dir, gt_dir):
    _setup_dir(RELEVANT_GLYPHS_MUSCIMA, HANDWRITTEN_FN_DATA)
    music_files = _get_music_files_from_dir(HANDWRITTEN_FN_XML)
    sorted_handwritten_files = sort_by_writers(music_files)
    content = dict()
    glyphs_per_score = []
    for music_file in sorted_handwritten_files:
        filename = 'CVC-MUSCIMA_W-' + str(music_file[0]).zfill(2) + '_N-' + \
                   str(music_file[1]).zfill(2) + '_D-ideal.xml'
        glyphs_per_score.append(muscima_score_ground_truth(filename, gt_dir))

    for glyphs_in_score in glyphs_per_score:
        for glyph in glyphs_in_score:
            if glyph.name not in content:
                content[glyph.name] = 1
            else:
                content[glyph.name] += 1

    glyph_count = dict((name, 0) for name in RELEVANT_GLYPHS_MUSCIMA)
    total_count = 0
    for i, (writer, page) in enumerate(sorted_handwritten_files):
        distortions = sorted_handwritten_files[(writer, page)]
        for file in distortions:
            file.position_glyphs(glyphs_per_score[i], content)
            img = cv.imread(join(images_dir, file.filename), cv.CV_8UC1)
            X, y = file.extract_training_data(img, content, RELEVANT_GLYPHS_MUSCIMA)
            _save_samples_to_disk(X, y, glyph_count, HANDWRITTEN_FN_DATA)
            total_count += 1
            print('{0} processed'.format(total_count))


def cross_section(first_coordinates, second_coordinates):
    assert first_coordinates[0] < first_coordinates[1] and second_coordinates[0] < second_coordinates[1]
    leftmost = first_coordinates
    if second_coordinates[0] > first_coordinates[0]:
        leftmost = second_coordinates

    rightmost = first_coordinates
    if second_coordinates[1] < first_coordinates[1]:
        rightmost = second_coordinates

    alpha = 0
    if rightmost[1] - leftmost[0] > 0:
        alpha = 1

    return (rightmost[1] - leftmost[0]) * alpha


def get_music_file_from_xml(fn):
    tree = ET.parse(fn)
    root = tree.getroot()
    assert root.tag == 'AutoScore'

    filename = root[0].text
    staff_height = int(root[1][0].text)
    staff_space = int(root[1][1].text)
    column = int(root[1][2].text)
    row = int(root[1][3].text)
    rot = float(root[1][4].text)
    model_gradient = [float(gradient) for gradient in str.split(root[1][5].text)]
    staffs = [int(staff.text) for staff in root[2]]

    return MusicFile(filename, staff_height, staff_space, column, row, rot, model_gradient, staffs)


def deepscores_score_ground_truth(score_filename, dataset_filename):
    tree = ET.parse(join(dataset_filename, splitext(score_filename)[0] + '.xml'))
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

        assert width and height
        name = node[0].text

        xmin, xmax = int(width * float(node[1][0].text)), int(width * float(node[1][1].text))
        ymin, ymax = int(height * float(node[1][2].text)), int(height * float(node[1][3].text))
        bbox = BBox(xmin, xmax, ymin, ymax)
        glyph = Glyph(name, bbox)
        glyphs.append(glyph)

    return glyphs


def muscima_score_ground_truth(score_filename, dataset_filename):
    tree = ET.parse(join(dataset_filename, splitext(score_filename)[0] + '.xml'))
    root = tree.getroot()
    assert root.tag == 'CropObjectList'

    glyphs = []
    for node in root[0]:
        name = node[1].text

        ymin = int(node[2].text)
        xmin = int(node[3].text)
        xmax = xmin + int(node[4].text)
        ymax = ymin + int(node[5].text)
        bbox = BBox(xmin, xmax, ymin, ymax)
        glyphs.append(Glyph(name, bbox))

    return glyphs


def sort_by_writers(music_files):
    sorted_files = dict()
    for file in music_files:
        pos = file.filename.find('p')
        page = int(file.filename[pos + 1: pos + 4])
        pos = file.filename.find('W-')
        if file.filename[pos + 3] != '_':
            writer = int(file.filename[pos + 2: pos + 4])
            pos += 4
        else:
            writer = int(file.filename[pos + 2])
            pos += 3

        if (writer, page) not in sorted_files:
            sorted_files[(writer, page)] = []

        distortion = file.filename[pos + 1:]
        file.filename = distortion + '/w-' + str(writer).zfill(2) + '/image/p' + str(page).zfill(3) + '.png'
        sorted_files[(writer, page)].append(file)

    return sorted_files


class MusicFile:
    def __init__(self, filename='', staff_height=0, staff_space=0,
                 column=0, row=0, rot=0, model_gradient=[], staff_starts=[]):
        self.filename = filename
        self.staff_height = staff_height
        self.staff_space = staff_space
        self.col = column
        self.row = row
        self.rot = rot
        self.model_gradient = model_gradient
        self.staff_starts = staff_starts
        self._compute_kernel()

    def position_glyphs(self, glyphs, dataset_content, minimum_occurences=POLL_THRESH):
        for glyph in glyphs:
            if glyph.name not in dataset_content:
                continue
            elif dataset_content[glyph.name] < minimum_occurences:
                continue

            x, y = self._get_avg_glyph_pos(glyph)
            closest_kernel_idx = self._get_closest_kernel_idx(x)
            if self._is_outside_of_staff_horizontal(closest_kernel_idx):
                continue

            closest_staff_idx, min_dist = self._get_closest_staff(y)
            if self._is_outside_of_staff_vertical(min_dist):
                continue

            self.glyphs_per_staff[closest_staff_idx][closest_kernel_idx].append(glyph)

    def extract_training_data(self, img, glyph_dict, relevant_glyphs,
                              area_overlap_thresh=AREA_OVERLAP_MIN):
        n_samples = self.n_divisions * len(self.glyphs_per_staff)
        X, y = [None] * n_samples, [None] * n_samples
        i = 0
        for staff_idx, staff in enumerate(self.glyphs_per_staff):
            for division_idx, division in enumerate(staff):
                division_box = self._get_division_bbox(staff_idx, division_idx)
                at_least_one_glyph = False
                max_poll = 0
                for glyph in division:
                    overlap = self._get_div_glyph_overlap(glyph, division_box)
                    area = (glyph.bbox.xmax - glyph.bbox.xmin) * (glyph.bbox.ymax - glyph.bbox.ymin)
                    if float(overlap) / (area + EPSILON) < area_overlap_thresh:
                        continue
                    poll = glyph_dict[glyph.name]
                    if poll > max_poll:
                        max_poll = poll
                        at_least_one_glyph = True
                div_img = self._get_img_bbox(img, division_box)
                label = 'None'
                if at_least_one_glyph:
                    if glyph.name not in relevant_glyphs:
                        label = 'Other'
                    else:
                        label = glyph.name
                X[i] = div_img
                y[i] = label
                i += 1
        return X, y

    def visualize_ground_truth(self, img, glyphs=True, divs=False, labels=True):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if divs:
            img = self._draw_divisions(img)
        if glyphs:
            for i, staff in enumerate(self.glyphs_per_staff):
                img = self._draw_staff(img, self.staff_starts[i])
                for division in staff:
                    for glyph in division:
                        img = self._draw_glyph(img, glyph)
                        if labels:
                            img = self._label_glyph(img, glyph, i)
        return img

    def _draw_staff(self, img, staff_pos):
        return cv.rectangle(img, (self.col, self.row + staff_pos),
                            (self.col + len(self.model_gradient), self.row + 5 * self.staff_height +
                             4 * self.staff_space + staff_pos), GREEN, 5)

    @staticmethod
    def _label_glyph(img, glyph, staff_idx):
        return cv.putText(img, glyph.name + ' ' + str(staff_idx), (glyph.bbox.xmin, glyph.bbox.ymax + 20),
                          cv.FONT_HERSHEY_SIMPLEX, 1, RED, 1, cv.LINE_AA)

    @staticmethod
    def _draw_glyph(img, glyph):
        return cv.rectangle(img, (glyph.bbox.xmin, glyph.bbox.ymin), (glyph.bbox.xmax, glyph.bbox.ymax), BLUE, 3)

    def _draw_divisions(self, img):
        for i in range(self.n_divisions):
            img = cv.line(img, (self.col + self.kernel_size[0] * i, 0),
                          (self.col + self.kernel_size[0] * i, self.height - 1), BLUE, 2)
        return img

    def _compute_kernel(self):
        assert self.staff_space and self.staff_height and len(self.model_gradient)
        self.kernel_size = (self.staff_space + 3 * self.staff_height, 11 * self.staff_height + 10 * self.staff_space)
        self.stride = int(self.kernel_size[0] * STRIDE_RATIO)
        self.n_strides_per_staff = int(len(self.model_gradient) / self.stride)
        self.boundary_adjust = BOUNDARY_EXTRA * (self.staff_height + self.staff_space)
        self.n_divisions = int(len(self.model_gradient) / self.kernel_size[0])
        self.glyphs_per_staff = [[[] for _ in range(self.n_divisions)] for _ in range(len(self.staff_starts))]

    def _get_closest_kernel_idx(self, x):
        return int(float(x) / len(self.model_gradient) * self.n_divisions)

    def _get_closest_staff(self, y):
        staff_centers = self._get_avg_staff_positions()
        closest_staff_index = 0
        min_dist = float('Inf')
        for i, pos in enumerate(staff_centers):
            dist = abs(y - pos)
            if dist < min_dist:
                min_dist = dist
                closest_staff_index = i
        return closest_staff_index, min_dist

    def _get_avg_staff_positions(self):
        return [start + 2 * (self.staff_space + self.staff_height) + int(self.staff_height / 2)
                for start in self.staff_starts]

    def _get_avg_glyph_pos(self, glyph):
        x = int((glyph.bbox.xmin + glyph.bbox.xmax) / 2) - self.col
        y = int((glyph.bbox.ymin + glyph.bbox.ymax) / 2) - self.row
        return x, y

    def _is_outside_of_staff_horizontal(self, x):
        return x >= self.n_divisions

    def _is_outside_of_staff_vertical(self, dist):
        return dist > 5 * (self.staff_height + self.staff_space)

    def _get_division_bbox(self, staff_idx, div_idx):
        xmin = div_idx * self.kernel_size[0] + self.col
        xmax = xmin + self.kernel_size[0]
        ymin = int(self.staff_starts[staff_idx] + self.row - (self.kernel_size[1] -
                                                              5 * self.staff_height - 4 * self.staff_space) / 2)
        ymax = ymin + self.kernel_size[1]
        return BBox(xmin, xmax, ymin, ymax)

    @staticmethod
    def _get_div_glyph_overlap(glyph, div_bbox):
        xmin, xmax, ymin, ymax = div_bbox
        overlap = cross_section((xmin, xmax), (glyph.bbox.xmin, glyph.bbox.xmax)) * cross_section(
            (ymin, ymax), (glyph.bbox.ymin, glyph.bbox.ymax))
        return overlap

    @staticmethod
    def _get_img_bbox(img, bbox):
        xmin, xmax, ymin, ymax = bbox
        return img[ymin:ymax, xmin:xmax]
