import torch
from torchvision import io
import cv2


def video_extract_opencv(file_path):
  cap = cv2.VideoCapture(file_path)
  while(True):
      ret, frame = cap.read()
      if ret == False:
        break
      cv2.imshow('frame',frame)
  cap.release()
  cv2.destroyAllWindows()



def video_extract_pytorch(file_path):
  vid = io.read_video(file_name)[0]
  vid_tensor = torch.Tensor(vid)
  #numpy array
  vid_arr = vid_tensor.numpy()
