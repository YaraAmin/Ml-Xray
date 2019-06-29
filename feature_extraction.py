import cv2
import numpy as np
from numpy import array

def corner detection(clear_img):
      maxCorners=200
      corners = cv2.goodFeaturesToTrack(img,maxCorners,0.01,2)
      corners = np.int0(corners)
      #Slice and Reshape Arrays for Machine Learning Model
      data = array(corners)
      x_image=np.array(corners)
      xx = x_image.reshape((x_image.shape[0], x_image.shape[2]))
      xxf=xx.reshape((x_image.shape[1],x_image.shape[0]*x_image.shape[2]))
      xxf=xx.reshape((x_image.shape[0]*x_image.shape[2]))
   #print("xxf",xxf,len(xxf),xxf.shape)
      if(len(xxf)==maxCorners*2):
            y.append(study_label[1])
            new_row= np.array(xxf)
            X = np.vstack([X, xxf]) if X.size else xxf
      return X

from skimage.feature import hog
def HOG(img):
      corners,corners_img = hog(img.reshape((256,256)), orientations=9, pixels_per_cell=(9, 9), cells_per_block=(1, 1), visualise=True)
      corners = np.int0(corners)
      data = array(corners)
      x_image=np.array(corners)
      xx = x_image.reshape((1, x_image.shape[0]))
      #print("xx", len(xx),xx.shape)
      if(len(xx)==1):
            new_row= np.array(xx)
            #print("ro",new_row.shape)
            #print("X",X.shape,X.size)
            X = np.vstack([X, xx]) if X.size else xx
      return X