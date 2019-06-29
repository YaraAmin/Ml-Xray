import cv2
import numpy as np
from numpy import array
from os import listdir
from os.path import isfile, join
from preprocessing import background_removal
import numpy as np
import cv2
from numpy import array
from os import listdir
from os.path import isfile, join
def read_image(path,case,study,i):
    img = []
    img_test = []
    img_label = []
    img_label_test = []
    y = []
    y_test = []
    X = np.array([])
    x_image = []
    ID = []
    X_test = np.array([])
    x_image_test = []
    ID_test = []
    train_positive = 0
    train_negative = 0
    test_positive = 0
    test_negative = 0
    image_path = path + case + '/' + study + '/' + i
    img = cv2.imread(image_path)
    study_label = study
    study_label = study_label.split('_')
    if (study_label[1] == 'positive'):
        study_label[1] = 1
        train_positive = train_positive + 1
    else:
        study_label[1] = 0
        train_negative = train_negative + 1
    return img, study_label[1]

def load_data(path,path2):
    
    X=np.array([])
    y=[]
    ID=[]
    X2=np.array([])
    y2=[]
    ID2=[]
    for case in listdir(path):
        for study in listdir(path+case+'/'):
            ID_image=case.split("patient")[1]
            image = [f for f in listdir (path+case+ '/'+study+'/') if isfile (join (path+case+ '/'+study+'/', f))]
            for i in image:
                img,label=read_image(path,case,study,i)
                #preprocessing
                clear_img=background_removal(img)
                vector=vectorize_image(clear_img)
                vector= np.array(vector)
                # X= featureExtraction(clear_img)
                X = np.vstack([X, vector]) if X.size else vector
                y.append(label)
                ID.append(ID_image)
    for case in listdir(path2):
        for study in listdir(path2+case+'/'):
            ID_image=case.split("patient")[1]
            image = [f for f in listdir (path2+case+ '/'+study+'/') if isfile (join (path2+case+ '/'+study+'/', f))]
            for i in image:
                img,label=read_image(path2,case,study,i)
                clear_img=background_removal(img)
                vector=vectorize_image(clear_img)
                vector= np.array(vector)
                # X2= featureExtraction(clear_img)
                X2 = np.vstack([X2, vector]) if X2.size else vector
                y2.append(label)
                ID2.append(ID_image)
    return X,y,ID,X2,y2,ID2

def vectorize_image(clear_img):
    vector = np.array(clear_img).flatten()
    vector= np.array(vector)
    #print("vector",vector)
    return vector
