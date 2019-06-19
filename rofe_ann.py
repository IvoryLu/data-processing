# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:37:38 2019

@author: 00098223
"""

#Part 1 - Data Preprocessing

##Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import csv
    #from imblearn.combine import SMOTEENN
    #import tensorflow as tf
    
def ann(i, epoch, maxmium, unit1):
    
    #Importing the libraries
    import numpy as np
    import pandas as pd
    import csv
    #Part 1 - Data Processing
    #Importing the dataset
    dataset = pd.read_csv('H:/Juan Lu/Data/Coxib/rofecoxib.csv')
    
    dataset = dataset.drop(["rootnum", "sup_date","match","rofecoxib","outcome","duration",
                             "death","duration_d","death.1","duration_d.1","duration",
                             "admission","duration_a","death_old","ACS","stroke.1","day",
                             "acs_stroke","cvd_death_updated"
                             ], axis=1)
    
    
    X = dataset.iloc[:,0:29].values
    y = dataset.iloc[:,29].values
    
    
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                          TomekLinks,
    #                                     NearMiss,
    #                                     InstanceHardnessThreshold,
    #                                     CondensedNearestNeighbour,
    #                                     EditedNearestNeighbours,
    #                                     RepeatedEditedNearestNeighbours,
    #                                     AllKNN,
    #                                     NeighbourhoodCleaningRule,
                                         OneSidedSelection)
    
    sampler = TomekLinks(random_state=42)
    #sampler = ClusterCentroids(random_state= 0)#slow
    #sampler = RandomUnderSampler(random_state=0)
    #sampler = NearMiss(version=1,random_state=0)
    #sampler = NearMiss(version=2,random_state=0)
    #sampler = NearMiss(version=3,random_state=0)
    #sampler = InstanceHardnessThreshold(random_state=0,
    #                                    estimator=LogisticRegression())
    #sampler = CondensedNearestNeighbour(random_state=0)#slow
    
    #sampler = OneSidedSelection(random_state=0)
    #sampler = NeighbourhoodCleaningRule(random_state=0)
    #sampler = EditedNearestNeighbours(random_state=0)
    ##sampler = RepeatedEditedNearestNeighbours(random_state=0)
    #sampler = AllKNN(random_state=0, allow_minority=True)
    
    #X_resample, y_resample = sampler.fit_sample(X, y)
    
    
    #Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size = 0.2, random_state = 42)
#    ran = i
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42 + 0)
    
    X_train, y_train = sampler.fit_sample(X_train, y_train)
    
    
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
                            
    
    #Part 2 - ANN
    #Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    
    #Initialising the ANN
    classifier = Sequential()
    
    #Adding the input layer and the first hidden layer 2250
    
    classifier.add(Dense(input_dim = 29,units= unit1, kernel_initializer = "uniform", activation= "relu"))
    
    #Adding the second hidden layer 825
    classifier.add(Dense(units = 825, kernel_initializer = "uniform", activation= "relu"))
    
    
    #Adding the third hidden layer 18
    classifier.add(Dense(units = 18, kernel_initializer = "uniform", activation= "relu"))#17
    
    
    #    #Adding the third hidden layer
    #    classifier.add(Dense(units = 18, kernel_initializer = "uniform", activation= "relu"))
    
    
    #Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation= "sigmoid"))
    #    
    #    
    #    import keras.backend as K
    #    
    #    def get_my_loss():
    #    
    #        def weighted_loss(y_true, y_pred):
    #            TP = K.sum(y_true * y_pred,axis = -1)
    #            FP = K.sum((1.- y_true)*y_pred,axis=-1)
    #            TN = K.sum((1.-y_true)*(1-y_pred),axis=-1)
    #            TP_2 = K.mean(K.sum((y_true - 1.) + y_pred, axis=-1))
    #            #P = K.sum(y_true)
    #            #F = K.sum(1. - y_true)
    #            FN = K.sum(y_true * (y_true - y_pred),axis= -1)
    #            FN_2 = K.mean(K.sum((1. - y_true)* (y_true - y_pred),axis=-1))
    #            return ( 0.26* FP - 12 * TP + 0.1 + 1.3 * K.mean(K.sum((1. - y_true)* (y_pred - y_true),axis=-1))) + FN + 0.35 * FN_2 - 0.05 * TN
    #            #return ( 0.076 * FP - 13 * TP + 0.10 + 1.61 * FN_2 + 2.81 * K.mean(K.sum((1. - y_true)* (y_pred - y_true),axis=-1))) - 0.05 * TN + 0.2 * FN
    #    
    #        return weighted_loss
    
#    import keras.backend as K
    
#    def pos_pre(y_true, y_pred):
#        TP = K.sum(y_true * y_pred,axis = -1)
#        FP = K.sum((1.- y_true)*y_pred,axis=-1)
#        TN = K.sum((1.-y_true)*(1-y_pred),axis=-1)
#        TP_2 = K.mean(K.sum((y_true - 1.) + y_pred, axis=-1))
#        FN = K.sum(y_true * (y_true - y_pred),axis= -1)
#        FN_2 = K.mean(K.sum((1. - y_true)* (y_true - y_pred),axis=-1))
#        
#        return 5*TP - FP + TN + TP_2 - FN - FN_2
#    
    #        err_II = y_true - pred
    #        #FN = tf.count_nonzero(FN)
    #        FN = tf.greater(err_II,0)
    #        FN = K.cast(FN,tf.float32)
    #        FN = tf.count_nonzero(FN)
    #        FN = K.cast(FN,tf.float32)
    #    
    #    
    #        err_I = pred - y_true
    #    
    #        #FP = tf.greater(err_I,0.3)
    #    
    #        FP = tf.greater(err_I,0)
    #        FP = K.cast(FP,tf.float32)
    #        FP = tf.count_nonzero(FN)
    #        FP = K.cast(FP,tf.float32)
    #    
    #        P = tf.count_nonzero(y_true)
    #        P = tf.maximum(P,1)
    #        P = tf.cast(P,tf.float32)
    #        print(P)
    #    
    #        M = tf.size(y_true)
    #        M = tf.cast(M,tf.float32)
    #        N = M - P
    #        print(N)
    #        N = tf.cast(N,tf.float32)
    #        fal_pos = FP/P
    
        #TP = tf.metrics.true_negatives(y_true,y_pred)
    #    return 1 - 1.2*FN/(N+FN)
    
    from keras import optimizers
    from keras.models import model_from_yaml
    opt = optimizers.Adam(lr= 0.001, beta_1=0.9, beta_2=0.999, epsilon=None,decay=0.0,amsgrad=False)
    
#    # load YAML and create model
#    yaml_file = open('model.yaml', 'r')
#    loaded_model_yaml = yaml_file.read()
#    yaml_file.close()
#    loaded_model = model_from_yaml(loaded_model_yaml)
#    # load weights into new model
#    loaded_model.load_weights("model.h5")
#    print("Loaded model from disk")
    
    #Compiling the ANNivo
    
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
#    loaded_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    #    classifier.compile(optimizer = opt, loss = get_my_loss(), metrics = ['accuracy'])
    
    
    #Fitting the ANN to the Training set
    #clf = make_pipeline(sampler,classifier)
    #clf.fit(X_train, y_train)
    
    #sample weight
    #sample_weight = np.ones((36288,))*0.5 + y_train*0.3#{0:1.,1:3.5}
    
    from sklearn.utils import class_weight
    sample_weight = class_weight.compute_sample_weight('balanced', y_train)
    #sample_weight = np.sqrt(sample_weight) 5 
    sample_weight = sample_weight**(i/2)
    
    from sklearn.utils import class_weight
    #1.4
    
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)
    class_weight = sample_weight**(100/2)
#    class_weight = {0:0.,
#                    1:0}
    #class_weight = {0: 1.,
    #                1: 19.3}
    
    #history = classifier.fit(X_train, y_train, batch_size=1400, epochs=15, validation_split=0.1, class_weight= class_weight) 
    history = classifier.fit(X_train, y_train, batch_size=epoch, epochs=10, validation_split=0.1, sample_weight = sample_weight) #  5600 10 class_weight= class_weight,
#    history = loaded_model.fit(X_train, y_train, batch_size=6000, epochs=10, validation_split=0.1, class_weight= class_weight, sample_weight = sample_weight)
    
    
    '''
    #Part 3 - Making the predictions and evaluating the model
    #Predicting the Test
    thre = 0.01
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > thre)
    
    #Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, y_pred)
    print('Roc auc score: ' + str(auc))
    
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report)
    
    dataset_c = pd.read_csv('H:\\Juan Lu\\Data\\Coxib\\temp\\cele_only.csv')
    y_c = dataset_c.iloc[:,36].values
    X_c = dataset_c.iloc[:,2:36].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size = 0.8, random_state = 42)
    X_train, y_train = sampler.fit_sample(X_train, y_train)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    history = classifier.fit(X_train, y_train, batch_size=1400, epochs=15, validation_split=0.1, class_weight= class_weight)
    
    dataset_i = pd.read_csv('H:\\Juan Lu\\Data\\Coxib\\temp\\ibup_only.csv')
    X_i = dataset_i.iloc[:,2:36].values
    y_i = dataset_i.iloc[:,36].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size = 0.8, random_state = 42)
    X_train, y_train = sampler.fit_sample(X_train, y_train)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    history = classifier.fit(X_train, y_train, batch_size=1400, epochs=3, validation_split=0.1, class_weight= class_weight)
    
    #file = open("data summary.txt", "a")
    #file.write('Roc auc score: ' + str(auc) +"\n" + 'threshold: '+str(thre)+ "\n")
    #file.write(report + "\n")
    #file.write('50,26,17,class_weight,1 - (FN + FP)/M ' + "\n")
    #file.close()
    
    
    score = 0
    score_max = 0
    thres = 0
    f1 = 0
    f1_max = 0
    auc_max = 0
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    threshold = 0.0
    precision_list = []
    recall_list = []
    for i in range (1,99):
        threshold = i/100
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > threshold)
        auc = roc_auc_score(y_test, y_pred)
        #print('Roc auc score: ' + str(auc))
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        score = precision[1] + recall[1]
        f1 = f1_score(y_test, y_pred)
        #print(str(threshold) + "," + str(f1))
        #print(score)
        if f1 > f1_max:
            f1_max = f1
            thres = threshold
            #score_max = score
            print('f1 score max: ' + str(f1_max))
            print('thres: ' + str(thres))
            print('precision + recall: ' + str(score))
            print('Roc auc score: ' + str(auc))
    
        if score > score_max:
            thres = threshold
            score_max = score
            print('f1 score: ' + str(f1))
            print('thres: ' + str(thres))
            print('precision + recall max : ' + str(score_max))
            print('Roc auc score: ' + str(auc))
    
        if auc > auc_max:
            auc_max = auc
            thres = threshold
            #score_max = score
            print('f1 score: ' + str(f1_max))
            print('thres: ' + str(thres))
            print('precision + recall: ' + str(score_max))
            print('Roc auc score max: ' + str(auc_max))
    
    #'''
            
    y_pred = classifier.predict(X_test)
#    y_pred = loaded_model.predict(X_test)
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
#        if roc_auc_score(y_test, y_pred) > maximum: 
#            # serialize model to YAML
#            model_yaml = classifier.to_yaml()
#            with open("model.yaml", "w") as yaml_file:
#                yaml_file.write(model_yaml)
#            # serialize weights to HDF5
#            classifier.save_weights("model.h5")
#            print("Saved model to disk")
            
    print(roc_auc_score(y_test, y_pred))
    
    total = []
    total.append(roc_auc_score(y_test, y_pred))
    total.append(epoch)
    total.append(i)
    
    with open('H:/Juan Lu/Data/Coxib/test_result.csv','a') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerows(total)    
    csvoutput.close()
    
    
    return roc_auc_score(y_test, y_pred)
     
score = [0]
#
for unit1 in range(57, 2250, 100)
for epoch in range(500, 3000, 500):
    for i in range(1,10):
        
        maximum = max(score)
        score.append(ann(i,epoch,maximum,unit1))

        



#
#plt.figure()
#lw = 1
#plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
#plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.show()
#
#
# summarize history for accuracy
#plt.plot(history.history['binary_accuracy'])
#plt.plot(history.history['val_binary_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
