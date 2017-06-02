import sc_detect


class Arg:
  def __init__(self):
    # manually set arguments
    self.downscale_factor = 2
    self.max_cal_width = 320
    self.max_cal_height = 300
    self.frame_skip = 1
    self.threshold = 30
    self.detection_method = "content"
    self.save_images = True
    self.thumb_width = 200
    self.thumb_path = None
    # other default values
    self.min_scene_len = 15
    self.min_percent = 95
    self.block_size = 32
    self.fade_bias = 0
    self.list_scenes = False
    self.quiet_mode = False
    self.start_time = None
    self.end_time = None
    self.duration = None
    self.stats_file = None


scene_detectors = sc_detect.detectors.get_available()
args = Arg()


def detect(filePath, format="FRAME", thumb_path=""):
  args.input = filePath
  args.thumb_path = thumb_path

  # Use above to initialize scene manager.
  smgr = sc_detect.manager.SceneManager(args, scene_detectors)
  # Scenes will be added to this list in detect_scenes().
  fps, frames_read, frames_processed = sc_detect.detect_scenes_file(filePath, smgr)

  # create new list with scene boundaries in milliseconds instead of frame #.
  scene_list_msec = [(1000.0 * x) / float(fps) \
    for x in smgr.scene_list]
  # create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
  scene_list_tc = [sc_detect.timecodes.get_string(x) \
    for x in scene_list_msec]

  if format == "MS":
    results = scene_list_msec
  elif format == "TC":
    results = scene_list_tc
  else:
    results = smgr.scene_list

  return fps, frames_read, results
