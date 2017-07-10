import cv2
import numpy as np
import os
import argparse
import logging

import vconfig
import scene_detect
import batch_im2txt

# ----------- configure logging ---------
log = logging.getLogger("vtr")
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# %(name)s -
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
ch.setFormatter(formatter)

log.addHandler(ch)
# ------  end of configure logging ---------

config = vconfig.VConfig()


def main():
  video = "testData/video.mp4"
  # video = "/Volumes/raid/work/2017.3/yoloDemo/testVideo.mp4"
  config.rootPath = os.path.dirname(os.path.realpath(__file__))
  thumb_path = os.path.join(config.rootPath, "thumbs/")
  config.thumb_path = thumb_path

  if not os.path.exists(thumb_path):
    os.mkdir(thumb_path)

  fps, frameCount, scenes, frames = scene_detect.detect(video,
    thumb_path=thumb_path)

  print("fps: {}, totalCount: {}, scenes: {}".format(fps, frameCount, len(frames)))
  print(scenes)

  # build tensorflow graph first
  batch_im2txt.build_graph(config.VOCABFILE, config.IM2TXT_CHECKPOINT_DIR)
  captions = batch_im2txt.parse(frames)


# def testim2txt():
#   frames = [cv2.imread("testData/2.jpg")]
#   batch_im2txt.build_graph(VOCABFILE, CHECKPOINT_DIR)
#   captions = batch_im2txt.parse(frames)


if __name__ == "__main__":
  main()
  # testim2txt()