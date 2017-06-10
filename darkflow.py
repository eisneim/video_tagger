import cv2
import sys
sys.path.insert(0, 'thirdparty/darkflow')
from darkflow.net.build import TFNet

tfnet = None

def build_net():
  options = {
    "model": "thirdparty/darkflow/cfg/yolo9000.cfg",
    "load": "thirdparty/darkflow/yolo9000.weights",
    "threshold": 0.3,
    "config": "thirdparty/darkflow/cfg",
  }

  tfnet = TFNet(options)
  print("======== finish build yolo9000 net =====")

def detect(imgs):
  global tfnet
  if not tfnet:
    build_net()

  results = []
  for img in imgs:
    rr = tfnet.return_predict(img)
    results.append(rr)
  return results


if __name__ == "__main__":
  img1 = cv2.imread("testData/ocr1.png")
  img2 = cv2.imread("testData/img2.jpg")
  results = detect([img1, img2])
  print(results)