import os


class VConfig:
  def __init__(self, config_path="config.conf"):
    if not os.path.exists(config_path):
      raise "config.conf does not found in root dir"

    self.IS_PRODUCTION = None
    self.max_cal_width = None
    self.max_cal_height = None
    self.max_pred_width = None
    self.max_pred_height = None
    self.frame_skip = 1
    self.TF_MODEL_FOLDER = None

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
        setattr(self, pair[0], pair[1])

    self.rootPath = os.path.dirname(os.path.realpath(__file__))
    self.thumb_path = os.path.join(self.rootPath, "thumbs/")



if __name__ == "__main__":
  config = VConfig()
