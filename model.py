
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model

# Get Resnet50 model from TF
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

print(tf.__version__)



class resnet:

    def __init__(self,num_classes,INIT_LR,NUM_EPOCHS,IMG_SHAPE):

        self.res_net50v2 = ResNet50V2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

        x = self.res_net50v2.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)

        self.predictions = Dense(num_classes, activation= 'softmax')(x)
        self.model = Model(inputs = self.res_net50v2.input, outputs = self.predictions)

        self.model.summary()

        # initialize the optimizer and compile the model
        print("[INFO] Initialize optimizer")
        self.opt = SGD(lr=INIT_LR,
                momentum=0.9,
                decay=INIT_LR / NUM_EPOCHS)

        self.model.compile(loss="categorical_crossentropy",
                    optimizer=self.opt,
                    metrics=["accuracy"])

    def train(self,BATCH_SIZE,NUM_EPOCHS, train_data,val_data):
        self.model.fit_generator(
        train_data,
        steps_per_epoch = train_data.samples // BATCH_SIZE,
        validation_data = val_data, 
        validation_steps = val_data.samples // BATCH_SIZE,
        epochs = NUM_EPOCHS, 
        verbose=1)
