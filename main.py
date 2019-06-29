import load_data
from load_data import load_data
from load_data import vectorize_image
from classification import RandomForestClassifier
from classification import KNN

from classification  import accuracy_per_image
from classification import accuracy_per_patient

import cv2
import numpy as np
from numpy import array
from os import listdir
from os.path import isfile, join

path_train ="MURA/XR_HAND/01 Fracture/train/"
path_test  ="MURA/XR_HAND/01 Fracture/test/"
data_train,label,ID,data_test,label_test,ID_test=load_data(path_train,path_test)
y_pred_RF=RandomForestClassifier(data_train,label,data_test)
y_pred_KNN=KNN(data_train,label,data_test)
print("accuracy_per_image_RF",accuracy_per_image(label_test,y_pred_RF))
accuracy_per_patient(ID_test,label_test,y_pred_RF)
print("accuracy_per_image_KNN",accuracy_per_image(label_test,y_pred_KNN))
accuracy_per_patient(ID_test,label_test,y_pred_KNN)