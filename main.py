import cv2
import numpy as np
import os
import argparse

import scene_detect
import batch_im2txt

VOCABFILE = "word_counts.txt"
CHECKPOINT_DIR = ("/Users/eisneim/www/deepLearning/"
  "_pre_trained_model/im2txt/model.ckpt-2000000")

def main():
  # video = "testData/video.mp4"
  video = "/Volumes/raid/work/2017.3/yoloDemo/testVideo.mp4"
  rootPath = os.path.dirname(os.path.realpath(__file__))
  thumb_path = os.path.join(rootPath, "thumbs/")
  if not os.path.exists(thumb_path):
    os.mkdir(thumb_path)

  fps, frameCount, scenes, frames = scene_detect.detect(video,
    thumb_path=thumb_path)

  print("fps: {}, totalCount: {}, scenes: {}".format(fps, frameCount, len(frames)))
  print(scenes)

  # build tensorflow graph first
  batch_im2txt.build_graph(VOCABFILE, CHECKPOINT_DIR)
  captions = batch_im2txt.describe(frames)


# def testim2txt():
#   frames = [cv2.imread("testData/2.jpg")]
#   batch_im2txt.build_graph(VOCABFILE, CHECKPOINT_DIR)
#   captions = batch_im2txt.describe(frames)


if __name__ == "__main__":
  main()
  # testim2txt()