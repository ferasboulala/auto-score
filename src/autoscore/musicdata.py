import cv2 as cv

import xml.etree.ElementTree as ET  
from collections import namedtuple
import os
from os.path import join

ARTIFICIAL_FN = '../datasets/Artificial/'
HANDWRITTEN_FN = '../datasets/Handwritten/'

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

RED = 0,0,255
BLUE = 255,0,0
GREEN = 0,255,0

def cross_section(first_coordinates, second_coordinates):
  '''
  Computes the overlap between two rectangles in R^2
  Guaranteed to work with python3 only
  '''
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
  def __init__(self, filename='', staff_height=0, staff_space=0, \
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
      
  # Positions every input glyph in the appropriate staff. For training only.
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
      # We ignore glyphs that stray too far from a set of staff lines
      if self._is_outside_of_staff_vertical(min_dist):
        continue
      
      self.glyphs_per_staff[closest_staff_idx][closest_kernel_idx].append(glyph)

  # Extracts images for every glyph. Returns an image and label. For training only.
  def extract_samples(self, img, glyph_dict, thresh=0.8):
    samples = []
    for staff_idx, staff in enumerate(self.glyphs_per_staff):
      for division_idx, division in enumerate(staff):
          xmin = division_idx * self.kernel_size[0] + self.col
          xmax = xmin + self.kernel_size[0]
          ymin = int(self.staff_starts[staff_idx] + self.row - (self.kernel_size[1] - \
                  5 * self.staff_height - 4 * self.staff_space) / 2)
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
          else:
            label = 'None'
          samples.append((div_img, label))
    return samples

  def visualize(self, img, glyphs=True, divs=False, labels=True):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    height, width, channels = img.shape
    if divs:
        img = self._draw_divisions(img)
    
    if glyphs :
        i = 0
        for staff in self.glyphs_per_staff:
            img = self._draw_staff(img, self.staff_starts[i])
            for division in staff:
                for glyph in division:
                    img = self._draw_glyph(img, glyph)
                    if labels:
                        img = self._label_glyph(img, glyph, i)
            i += 1
    return img

  def _draw_staff(self, img, staff_pos):
    return cv.rectangle(img, (self.col, self.row + staff_pos), \
                            (self.col + len(self.model_gradient), self.row + 5 * self.staff_height + \
                            4 * self.staff_space + staff_pos), GREEN, 5)

  def _label_glyph(self, img, glyph, i):
    return cv.putText(img, glyph.name + ' ' + str(i), (glyph.bbox.xmin, glyph.bbox.ymax + 20), \
                      cv.FONT_HERSHEY_SIMPLEX, 1, RED, 1, cv.LINE_AA)

  def _draw_glyph(self, img, glyph):
    return cv.rectangle(img, (glyph.bbox.xmin, glyph.bbox.ymin), (glyph.bbox.xmax, glyph.bbox.ymax), RED, 3)

  def _draw_divisions(self, img):
    for i in range(self.n_divisions):
      img = cv.line(img, (self.col + self.kernel_size[0] * i, 0), \
      (self.col + self.kernel_size[0] * i, height - 1), BLUE, 2)
    return img

  def _compute_kernel(self):
    assert self.staff_space and self.staff_height and len(self.model_gradient)
    self.kernel_size = (self.staff_space + 3 * self.staff_height, 11 * self.staff_height + 10 * self.staff_space)   
    self.stride = int(self.kernel_size[0] * STRIDE_RATIO)
    self.n_strides_per_staff = int(len(self.model_gradient ) / self.stride)
    self.boundary_adjust = BOUNDARY_EXTRA * (self.staff_height + self.staff_space)
    self.n_divisions = int(len(self.model_gradient) / self.kernel_size[0])
    self.glyphs_per_staff = [[[] for j in range(self.n_divisions)] for  i in range(len(self.staff_starts))]

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
          closest_staff_idx = i
    return closest_staff_idx, min_dist

  def _get_avg_staff_positions(self):
    return [start + 2 * (self.staff_space + self.staff_height) + int(self.staff_height / 2) \
            for start in self.staff_starts]
    
  def _get_avg_glyph_pos(self, glyph):
    x = int((glyph.bbox.xmin + glyph.bbox.xmax) / 2) - self.col
    y = int((glyph.bbox.ymin + glyph.bbox.ymax) / 2) - self.row
    return x, y

  def _is_outside_of_staff_horizontal(self, x):
    return x >= self.n_divisions

  def _is_outside_of_staff_vertical(self, dist):
    return dist > 5 * (self.staff_height + self.staff_space)

def get_music_file_from_xml(fn):
  '''
  Parses the staff xml annotation and returns a MusicFile object
  '''
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
  '''
  Parses the deepscores groundtruth and returns an array of glyphs
  '''
  tree = ET.parse(join(dataset_filename, os.path.splitext(score_filename)[0] + '.xml'))
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
