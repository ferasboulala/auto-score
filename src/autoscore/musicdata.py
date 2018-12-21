import cv2 as cv
import xml.etree.ElementTree as ET  
from collections import namedtuple
from os.path import join, splitext, isfile
from os import listdir

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
POLL_THRESH = 50
# Useful constant
EPSILON = 1e-7
# Area overlap
AREA_OVERLAP_MIN = 0.6

RED = 255,0,0
BLUE = 0,0,255
GREEN = 0,255,0

RELEVANT_GLYPHS = set(
  [
  'noteheadBlack',
  'noteheadHalf',
  'gClef',
  'keySharp',
  'accidentalNatural',
  'noteheadWhole',
  'accidentalSharp',
  'g-clef',
  'sharp',
  'notehead-full',
  'flat',
  'natural',
  'notehead-empty',
  'None' 
  ]
)

def get_deepscores_data(images_dir, gt_dir):
  filenames = [f for f in listdir(ARTIFICIAL_FN) if isfile(join(ARTIFICIAL_FN, f))]
  music_files = [get_music_file_from_xml(join(ARTIFICIAL_FN, f)) for f in filenames]
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
  
  X, y = [], []
  for i, music_file in enumerate(music_files):
    music_file.position_glyphs(glyphs_per_score[i], content)
    img = cv.imread(join(images_dir, music_file.filename), cv.CV_8UC1)
    x, labels = music_file.extract_training_data(img, content, RELEVANT_GLYPHS)
    X.extend(x)
    y.extend(labels)

  return X, y

# TODO : Do the equivalent of get_deepscores_data for the muscima dataset
def get_muscima_data(images_dir, gt_dir):
  pass

def cross_section(first_coordinates, second_coordinates):
  '''
  Computes the overlap between two rectangles in R^2
  '''
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
  tree = ET.parse(join(dataset_filename, splitext(score_filename)[0] + '.xml'))
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
  '''
  Sorts MusicFile objects by writer and page for the muscima dataset
  '''
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

class MusicFile:
  '''
  Class that holds information about a music sheet
  '''
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
      if self._is_outside_of_staff_vertical(min_dist):
        continue
      
      self.glyphs_per_staff[closest_staff_idx][closest_kernel_idx].append(glyph)

  # Extracts images for every glyph. Returns an image and label. For training only.
  def extract_training_data(self, img, glyph_dict, relevant_glyphs=RELEVANT_GLYPHS, area_overlap_thresh=AREA_OVERLAP_MIN):
    n_samples = self.n_divisions * len(self.glyphs_per_staff)
    X, y = [None] * n_samples , [None] * n_samples
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

  def visualize(self, img, glyphs=True, divs=False, labels=True):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    height, width, channels = img.shape
    if divs:
        img = self._draw_divisions(img)
    if glyphs :
      for i, staff in enumerate(self.glyphs_per_staff):
        img = self._draw_staff(img, self.staff_starts[i])
        for division in staff:
          for glyph in division:
            img = self._draw_glyph(img, glyph)
            if labels:
              img = self._label_glyph(img, glyph, i)
    return img

  def _draw_staff(self, img, staff_pos):
    return cv.rectangle(img, (self.col, self.row + staff_pos), \
                            (self.col + len(self.model_gradient), self.row + 5 * self.staff_height + \
                            4 * self.staff_space + staff_pos), GREEN, 5)

  def _label_glyph(self, img, glyph, staff_idx):
    return cv.putText(img, glyph.name + ' ' + str(staff_idx), (glyph.bbox.xmin, glyph.bbox.ymax + 20), \
                      cv.FONT_HERSHEY_SIMPLEX, 1, RED, 1, cv.LINE_AA)

  def _draw_glyph(self, img, glyph):
    return cv.rectangle(img, (glyph.bbox.xmin, glyph.bbox.ymin), (glyph.bbox.xmax, glyph.bbox.ymax), BLUE, 3)

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

  def _get_division_bbox(self, staff_idx, div_idx):
    xmin = div_idx * self.kernel_size[0] + self.col
    xmax = xmin + self.kernel_size[0]
    ymin = int(self.staff_starts[staff_idx] + self.row - (self.kernel_size[1] - \
            5 * self.staff_height - 4 * self.staff_space) / 2)
    ymax = ymin + self.kernel_size[1]
    return BBox(xmin, xmax, ymin, ymax)

  def _get_div_glyph_overlap(self, glyph, div_bbox):
    xmin, xmax, ymin, ymax = div_bbox
    overlap = cross_section((xmin, xmax), (glyph.bbox.xmin, glyph.bbox.xmax)) * cross_section(
              (ymin, ymax), (glyph.bbox.ymin, glyph.bbox.ymax))
    return overlap

  def _get_img_bbox(self, img, bbox):
    xmin, xmax, ymin, ymax = bbox
    return img[ymin:ymax, xmin:xmax]