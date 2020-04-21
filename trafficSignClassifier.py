import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model


################# Parameters #####################
path = "/home/utkarsh/Downloads/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images" # folder with all the test images
path1 = "/home/utkarsh/Downloads/GTSRB-Training_fixed/GTSRB"
batch_size_val=50  # how many to process together
steps_per_epoch_val=2000
epochs_val=15
imageDimesions = (32,32,3)
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
noOfClasses = 43
###################################################

def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
                         # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2=(3,3)
    size_of_pool=(2,2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_Of_Nodes = 500   # NO. OF NODES IN HIDDEN LAYERS
    model= Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(imageDimesions[0],imageDimesions[1],3),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax')) # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())


def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)      # CONVERT TO GRAYSCALE
    #img = cv2.equalizeHist(img)                    # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255                                   # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    img = np.dstack([img, img, img])
    return img


datagen = ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10,
                            validation_split = validationRatio,
                            preprocessing_function = preprocessing)



train_generator = datagen.flow_from_directory('/home/utkarsh/Downloads/GTSRB-Training_fixed/GTSRB/Training',
                                              subset = 'training',
                                              target_size=(32,32),
                                              batch_size=200)



val_generator = datagen.flow_from_directory('/home/utkarsh/Downloads/GTSRB-Training_fixed/GTSRB/Training',
                                              subset = 'validation',
                                              target_size=(32,32),
                                              batch_size=200)



history = model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch_val,
                            epochs=epochs_val,
                            validation_data=val_generator,
                            shuffle=1)

########################################################
# SAVE THE MODEL THE FIRST TIME YOU TRAIN IT

#model.save('/home/utkarsh/Desktop/trafficSigns.h5')
########################################################


########################################################
# LOAD MODEL ONCE IT HAS BEEN SAVED OR LOAD MY MODEL 

#model = load_model('/home/utkarsh/Desktop/trafficSigns.h5')
########################################################



########################################################
# PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
########################################################



########################################################
# LOAD THE TEST IMAGES
images = []
names = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for y in myList:
    names.append(y)
    curImg = cv2.imread(path+"/"+y)
    curImg = cv2.resize(curImg, (32, 32))
    img = cv2.cvtColor(curImg,cv2.COLOR_BGR2GRAY)
    img = img/255
    img = np.dstack([img, img, img])
    images.append(img)
images = np.array(images)
########################################################



########################################################
# GET THE ANSWER LABELS FROM THE CSV 
finalAns = []
df1 = pd.read_csv('/home/utkarsh/Downloads/GTSRB_Final_Test_Images/GTSRB/Final_Test/GT-final_test.csv', sep=';')
for name in names:
    df2 = df1[df1['Filename']==name]
    finalAns.append(int(df2['ClassId'].values))
########################################################



########################################################
# MAKE PREDICTIONS
pred = model.predict_classes(images)
########################################################



########################################################
# CONFUSION MATRIX AND CLASSIFICATION REPORT
mat = confusion_matrix(finalAns, pred)
clfrpt = classification_report(finalAns, pred)
########################################################



########################################################
# MAKE A HEATPAN FOR BETTER VISUALISATION
sns.heatmap(mat, cmap='winter')
########################################################