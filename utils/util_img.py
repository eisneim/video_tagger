from datetime import datetime
from os import mkdir
from os.path import exists, join
import cv2


def ratio_scale_factor(width, height, maxw, maxh):
  """scale image while keep ratio

  according to max width and max height, decide proper dest
  width and height, return zoom factor

  Arguments:
    width {number} -- original width
    height {number} -- original height
    maxw {number} -- maxium dest width
    maxh {number} -- maxium dest image height

  Returns:
    zoomfactor[number] -- always less than 1, dest / original
  """
  zoomfactor = 1
  ratio = width / height
  setedRatio = maxw / maxh
  # if image size is too big, resize it to fit window size and display
  destw = desth = 0
  if ratio >= setedRatio and width >= maxw:
    destw = maxw
    desth = maxw / ratio
    # print("landscape: {}x{} => {}x{}".format(width, height, destw, desth))
  elif ratio < setedRatio and height >= maxh:
    desth = maxh
    destw = maxh * ratio
    # print("portrait: {}x{} => {}x{}".format(width, height, destw, desth))
  else:
    # no resize required
    zoomfactor
  zoomfactor = destw / width
  return zoomfactor


def save_thumbs(videoName, imgs, frameNumbers, thumbPath):
  monthStr = datetime.today().strftime("%Y-%m")
  destFolder = join(thumbPath, monthStr)
  if not exists(destFolder):
    mkdir(destFolder)

  for idx, img in enumerate(imgs):
    framNum = frameNumbers[idx]
    imgpath = join(destFolder, "{}_({}).jpg".format(videoName, framNum))
    cv2.imwrite(imgpath, img)

