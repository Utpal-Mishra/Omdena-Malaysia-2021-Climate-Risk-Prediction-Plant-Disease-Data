from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense#, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224, 224]

train_path = 'DiseaseData/train'
valid_path = 'DiseaseData/val'

model = DenseNet201(include_top = False, 
            weights = "imagenet", 
            input_shape = IMAGE_SIZE + [3])

for layer in model.layers:
    layer.trainable = False


folders = glob("DiseaseData/train/*")

x = Flatten()(model.output)

prediction = Dense(len(folders), activation = "softmax")(x)

model = Model(inputs = model.input, outputs = prediction)

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('DiseaseData/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32, 
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('DiseaseData/test',
                                            target_size = (224, 224),
                                            batch_size = 32, 
                                            class_mode = 'categorical')

r = model.fit_generator(
    training_set,
    validation_data = test_set,
    epochs = 30,
    steps_per_epoch = len(training_set),
    validation_steps = len(test_set))

data = pd.read_csv("ModelPerformance.csv")[['Models', 'TA_Epoch_30', 'TE_Epoch_30']]
print(data)

temp = pd.DataFrame({"Models": ["DenseNet201"],
                     "TA_Epoch_30": model.evaluate_generator(test_set, 400)[1],
                     "TE_Epoch_30": model.evaluate_generator(test_set, 400)[0]})
  
temp = pd.concat([data, temp], ignore_index=True)[['Models', 'TA_Epoch_30', 'TE_Epoch_30']]

print(temp[['Models', 'TA_Epoch_30', 'TE_Epoch_30']])
temp.to_csv("ModelPerformance.csv")

# LOSS
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = "val loss")
plt.legend()
plt.show()
plt.savefig("LossVal_loss")

# ACCURACY
plt.plot(r.history['accuracy'], label = 'train acc')
plt.plot(r.history['val_accuracy'], label = "val acc")
plt.legend()
plt.show()
plt.savefig("AccVal_acc")

import tensorflow as tf

from tensorflow.keras.models import load_model

model.save('model_densenet201.h5')
