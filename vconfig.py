import os


class VConfig:
  def __init__(self, config_path="config.conf"):
    if not os.path.exists(config_path):
      raise "config.conf does not found in root dir"

    self.PARALLEL = False
    self.THUMB_DIR = None
    self.IS_PRODUCTION = None
    self.max_cal_width = None
    self.max_cal_height = None
    self.max_pred_width = None
    self.max_pred_height = None
    self.frame_skip = 1
    self.TF_MODEL_FOLDER = None
    self.THRESHOLD_OBJECT_DETECTION = 0.6
    self.THRESHOLD_SCENESEG = 30
    self.SCENESEG_MINFRAMES = 15

    # config_dict = {}
    with open(config_path, "r") as ff:
      for line in ff:
        line = line.strip("\n")
        if not line:
          continue

        pair = line.split(":=")
        # make sure the value doesn't contains ":="
        assert len(pair) == 2
        # config_dict[pair[0]] = config_dict[pair[1]]
        print("{} = {}".format(pair[0], pair[1]))
        if pair[1] == "True":
          setattr(self, pair[0], True)
        elif pair[1] == "False":
          setattr(self, pair[0], False)
        elif pair[1] == "None":
          setattr(self, pair[0], None)
        else:
          setattr(self, pair[0], pair[1])

    self.rootPath = os.path.dirname(os.path.realpath(__file__))
    self.thumb_path = os.path.join(self.rootPath, "thumbs/")

    if self.max_cal_width:
      self.max_cal_width = int(self.max_cal_width)
    if self.max_cal_height:
      self.max_cal_height = int(self.max_cal_height)
    if self.max_pred_width:
      self.max_pred_width = int(self.max_pred_width)
    if self.max_pred_height:
      self.max_pred_height = int(self.max_pred_height)


if __name__ == "__main__":
  config = VConfig()
