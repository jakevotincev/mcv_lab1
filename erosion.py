# ITMO University
# Mobile Computer Vision course
# 2020
# by Aleksei Denisov
# denisov@itmo.ru

import cv2
import numpy as np
import time

def erode(image_file, erosion_level=6):

    structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)

    orig_shape = image_file.shape
    pad_width = erosion_level - 2

    # pad the matrix with `pad_width`
    image_pad = np.pad(array=image_file, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    # sub matrices of kernel size
    flat_submatrices = np.array([
        image_pad[i:(i + erosion_level), j:(j + erosion_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    # condition to replace the values - if the kernel equal to submatrix then 255 else 0
    image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in flat_submatrices], dtype=np.uint8)
    image_erode = image_erode.reshape(orig_shape)

    return image_erode

def main():
  cap = cv2.VideoCapture("10s.mp4")

  total_frames = 0
  start_time = time.time()
  if cap.isOpened():
      while True:
          ret_val, frame = cap.read()

          if not ret_val:
              break

          grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          ret, grayframe = cv2.threshold(grayframe, 127, 255, cv2.THRESH_BINARY)

          kernel = np.ones((12,12), np.uint8)

          # cv2_ero = cv2.erode(grayframe, kernel, iterations=1)
          my_ero = erode(grayframe)

          # cv2.imshow('CV2_ERO', cv2_ero)
          cv2.imshow('MY_ERO', my_ero)

          # This also acts as
          keyCode = cv2.waitKey(1) & 0xFF
          # Stop the program on the ESC key
          if keyCode == 27:
              break
          total_frames += 1

      end_time = time.time()
      fps = total_frames / (end_time - start_time)
      print(f"Frames per second: {fps}")
      print(f"Total frames: {total_frames}")
      print(f"Duration: {end_time-start_time}")

if __name__ == "__main__":
   main()

