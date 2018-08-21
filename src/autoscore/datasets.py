import cv2 as cv

import xml.etree.ElementTree as ET  
from collections import namedtuple
import os
from os.path import join

# Musical symbol called a glyph
Glyph = namedtuple('Glyph', 'name bbox')

# Bounding box of an entity in an image
BBox = namedtuple('Bbox', 'xmin xmax ymin ymax')

# Portion of the kernel that is exclusive to the previous one
STRIDE_RATIO = 0.5
# Amount of staff space added over the staff lines
BOUNDARY_EXTRA = 4
# Minimum amount of samples for training
POLL_THRESH = 200
# Useful constant
EPSILON = 1e-7

'''
Function that computes the overlap between two segments
Guaranteed to work with python3 only
'''
def cross_section(first_coordinates, second_coordinates):
    assert first_coordinates[0] < first_coordinates[1] and second_coordinates[0] < second_coordinates[1]  
    leftmost = first_coordinates
    if second_coordinates[0] > first_coordinates[0]:
        leftmost = second_coordinates
    
    rightmost = first_coordinates
    if second_coordinates[1] < first_coordinates[1]:
        rightmost = second_coordinates
    
    return (rightmost[1] - leftmost[0]) * (rightmost[1] - leftmost[0] > 0)

'''
Class that holds information about a music sheet
'''
class MusicFile:
    def __init__(self, filename='', height=0, space=0, column=0, row=0, rot=0, model=[], staff_start=[]):
        self.filename = filename
        self.height = height
        self.space = space
        self.col = column
        self.row = row
        self.rot = rot
        self.model = model
        self.staff_start = staff_start        
    
    # Computes the size of the kernel that will move on the image
    def compute_kernel(self):
        assert self.space and self.height and len(self.model)
        self.kernel_size = (self.space + 3 * self.height, 11 * self.height + 10 * self.space)   
        self.stride = int(self.kernel_size[0] * STRIDE_RATIO)
        self.n_strides_per_staff = int(len(self.model) / self.stride)
        self.boundary_adjust = BOUNDARY_EXTRA * (self.height + self.space)
        self.n_divisions = int(len(self.model) / self.kernel_size[0])
        
    # Positions every input glyph in the appropriate staff. For training only.
    def locate(self, glyphs, glyph_dict, thresh=POLL_THRESH):
        staff_pos = [staff + 2 * (self.space + self.height) + int(self.height / 2) for staff in self.staff_start]
        self.staffs = [[[] for j in range(self.n_divisions)] for  i in range(len(self.staff_start))]
        for glyph in glyphs:
            if glyph.name not in glyph_dict:
                continue
            elif glyph_dict[glyph.name] < POLL_THRESH:
                continue
            
            x = int((glyph.bbox.xmin + glyph.bbox.xmax) / 2) - self.col
            y = int((glyph.bbox.ymin + glyph.bbox.ymax) / 2) - self.row
            
            closest_kernel_idx = int(float(x) / len(self.model) * self.n_divisions)
            # We ignore glyphs outside the staff model in column position
            if closest_kernel_idx >= self.n_divisions:
                continue
            
            min_dist = float('Inf')
            closest_staff_idx = 0
            for i in range(len(staff_pos)):
                dist = abs(y - staff_pos[i])
                if min_dist > dist:
                    min_dist = dist
                    closest_staff_idx = i
                    
            if min_dist > 5 * (self.height + self.space):
                continue
            
            self.staffs[closest_staff_idx][closest_kernel_idx].append(glyph)
    
    # Extracts images for every glyph. Returns an image and label. For training only.
    def extract_samples(self, img, glyph_dict, thresh=0.8):
        samples = []
        for staff_idx, staff in enumerate(self.staffs):
            prev_glyph = None
            for division_idx, division in enumerate(staff):
                xmin = division_idx * self.kernel_size[0] + self.col
                xmax = xmin + self.kernel_size[0]
                ymin = int(self.staff_start[staff_idx] + self.row - (self.kernel_size[1] - 5 * self.height - 4 * self.space) / 2)
                ymax = ymin + self.kernel_size[1]
                division_box = BBox(xmin, xmax, ymin, ymax)
                polled_glyph = None
                max_poll = 0
                for glyph in division:
                    # Area filtering
                    overlap = cross_section((xmin, xmax), (glyph.bbox.xmin, glyph.bbox.xmax)) * cross_section(
                        (ymin, ymax), (glyph.bbox.ymin, glyph.bbox.ymax))
                    area = (glyph.bbox.xmax - glyph.bbox.xmin) * (glyph.bbox.ymax - glyph.bbox.ymin)
                    assert overlap <= area
                    if float(overlap) / (area + EPSILON) < thresh:
                        continue
                    # Poll filtering
                    poll = glyph_dict[glyph.name]
                    if poll > max_poll:
                        max_poll = poll
                        polled_glyph = glyph
                
                div_img = img[ymin:ymax, xmin:xmax]
                if polled_glyph is not None:
                    label = glyph.name
                    prev_glyph = polled_glyph
                else:
                    label = 'None'
                    prev_glyph = None
                samples.append((div_img, label))
        return samples
    
    # Annotates the input image.
    def visualize(self, img, glyphs=True, divs=True, labels=True):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        height, width, channels = img.shape
        if divs:
            for i in range(self.n_divisions):
                img = cv.line(img, (self.col + self.kernel_size[0] * i, 0), 
                         (self.col + self.kernel_size[0] * i, height - 1), (0,0,255), 2)
        
        if glyphs :
            i = 0
            for staff in self.staffs:
                img = cv.rectangle(img, (self.col, self.row + self.staff_start[i]), 
                                   (self.col + len(self.model), self.row + 5 * self.height + 4 * self.space + 
                                    self.staff_start[i]), (0,255,0), 5)
                for division in staff:
                    
                    for glyph in division:
                        img = cv.rectangle(img, 
                        (glyph.bbox.xmin, glyph.bbox.ymin), (glyph.bbox.xmax, glyph.bbox.ymax), (255,0,0), 3)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        if labels:
                            img = cv.putText(img, glyph.name + ' ' + str(i), (glyph.bbox.xmin, glyph.bbox.ymax + 20), 
                                          font, 1, (0,0,255), 1, cv.LINE_AA)
                i += 1
        return img

'''
Parses the staff xml annotation and returns a MusicFile object
'''
def staff_xml(fn):
    tree = ET.parse(fn)
    root = tree.getroot()
    assert root.tag == 'AutoScore'
    
    filename = root[0].text
    height = int(root[1][0].text)
    space = int(root[1][1].text)
    column = int(root[1][2].text)
    row = int(root[1][3].text)
    rot = float(root[1][4].text)
    model = [float(gradient) for gradient in str.split(root[1][5].text)]
    staffs = [int(staff.text) for staff in root[2]]

    return MusicFile(filename, height, space, column, row, rot, model, staffs)

def blahblahblah():
    print('hi')
    return

'''
Parses the deepscores groundtruth and returns an array of glyphs
'''
def deepscores_gt(filename, deepscores_gt_fn, deepscores_dict):
    tree = ET.parse(join(deepscores_gt_fn, os.path.splitext(filename)[0] + '.xml'))
    root = tree.getroot()
    assert root.tag == 'annotation'
    
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

        if name not in deepscores_dict:
            deepscores_dict[name] = 0
        deepscores_dict[name] += 1
        
        xmin, xmax = int(width * float(node[1][0].text)), int(width * float(node[1][1].text))
        ymin, ymax = int(height * float(node[1][2].text)), int(height * float(node[1][3].text))
        bbox = BBox(xmin, xmax, ymin, ymax)
        glyph = Glyph(name, bbox)
        glyphs.append(glyph)
            
    return glyphs

'''
Parses the muscima groundtruth and returns an array of glyphs
'''
def muscima_gt(filename, muscima_gt_fn, muscima_dict):
    tree = ET.parse(join(muscima_gt_fn, os.path.splitext(filename)[0] + '.xml'))
    root = tree.getroot()
    assert root.tag == 'CropObjectList'
    
    # For every glyph
    glyphs = []
    for node in root[0]:
        name = node[1].text
        
        if name not in muscima_dict:
            muscima_dict[name] = 0
        muscima_dict[name] += 1
        
        ymin = int(node[2].text)
        xmin = int(node[3].text)
        xmax = xmin + int(node[4].text)
        ymax = ymin + int(node[5].text)
        bbox = BBox(xmin, xmax, ymin, ymax)
        glyphs.append(Glyph(name, bbox))
        
    return glyphs


'''
Sorts MusicFile objects by writer and page for the muscima dataset
'''
def sort_by_writers(music_files):
    sorted_files = dict()
    for file in music_files:
        pos = file.filename.find('p')
        page = int(file.filename[pos + 1 : pos + 4])
        pos = file.filename.find('W-')
        if (file.filename[pos + 3] != '_'):
            writer = int(file.filename[pos + 2 : pos + 4])
            pos += 4
        else:
            writer = int(file.filename[pos + 2])
            pos += 3
            
        if (writer,page) not in sorted_files:
            sorted_files[(writer,page)] = []
        
        distortion = file.filename[pos + 1:]
        file.filename = distortion + '/w-' + str(writer).zfill(2) + '/image/p' + str(page).zfill(3) + '.png'
            
        sorted_files[(writer, page)].append(file)
    
    return sorted_files