import numpy as np

def RandomForestClassifier(X,y,X_test):
    from sklearn.ensemble import RandomForestClassifier
    svclassifier = RandomForestClassifier(n_estimators=100 ,criterion='entropy')
    #fit to the trainin data
    svclassifier.fit(X, y)
    # predict the value of test data
    y_pred = svclassifier.predict(X_test)
    return y_pred

def KNN(X,y,X_test):
    from sklearn.neighbors import KNeighborsClassifier  
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X, y) 
    y_pred = classifier.predict(X_test)  
    return y_pred

def SVM(X,y,X_test):
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='rbf',random_state=0)  #Gaussian Kernel
    #fit to the trainin data
    svclassifier.fit(X, y)
    # predict the value of test data
    y_pred = svclassifier.predict(X_test)
    return y_pred

def accuracy_per_image(y_test,y_pred):
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    false_negative = confusion_matrix(y_test, y_pred)[0][0]
    true_positive = confusion_matrix(y_test, y_pred)[1][1]
    total = (confusion_matrix(y_test, y_pred)[0][0])+(confusion_matrix(y_test, y_pred)[0][1])+(confusion_matrix(y_test, y_pred)[1][0])+(confusion_matrix(y_test, y_pred)[1][1])
    accuracy = ((true_positive+false_negative)/total)*100
    print("accuracy per image",accuracy)
    return

def accuracy_per_patient(ID_test,y_test,y_pred):
    index_list=[]
    patient_number_test=[]
    patient_number_pred=[]
    count = 0
    for i in range(0 , len(ID_test)):
       if i not in index_list :
           patient_number_test.append(0)
           patient_number_pred.append(0)
           the_patient = ID_test[i]
           small_index_list = []
           for j in range (i ,len(ID_test)):
                if ID_test[j] == the_patient :
                    small_index_list.append(j)
                    index_list.append(j)
           for h in small_index_list:
                if int(y_test[h]) == 1 :
                    patient_number_test[count] = 1
                if int (y_pred[h]) == 1 :
                    patient_number_pred[count] = 1
           count =count +1
    # print (patient_number_test)
    # print(patient_number_pred)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(patient_number_test, patient_number_pred))
    print(classification_report(patient_number_test, patient_number_pred))
    false_negative_patient = confusion_matrix(patient_number_test, patient_number_pred)[0][0]
    true_positive_patient = confusion_matrix(patient_number_test, patient_number_pred)[1][1]
    total_patient = (confusion_matrix(patient_number_test, patient_number_pred)[0][0])+(confusion_matrix(patient_number_test, patient_number_pred)[0][1])+(confusion_matrix(patient_number_test, patient_number_pred)[1][0])+(confusion_matrix(patient_number_test, patient_number_pred)[1][1])
    accuracy_patient = ((true_positive_patient+false_negative_patient)/total_patient)*100
    print("accuracy per patient",accuracy_patient)
    return
