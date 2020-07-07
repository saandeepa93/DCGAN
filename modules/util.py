from skimage import io
import cv2
import yaml


def ioshow(img):
  io.imshow(img)
  io.show()

def cvshow(img):
  cv2.imshow("img", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def get_config():
  with open('./config.yaml') as file:
    configs = yaml.load(file, Loader = yaml.FullLoader)
  return configs