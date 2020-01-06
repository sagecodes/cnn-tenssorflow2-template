from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Train_generator:

    """
    TODO: Add doc string
    """

    def __init__(self,img_height,img_width,batch_size, data_dir):
        
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
        #     shear_range=0.2,
        #     zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)

    def image_load_csv(self,df):

        # Shuffle dataFrame 
        Shuffle_file_paths_and_labels_df = df.sample(frac=1).reset_index(drop=True)

        self.train_generator = self.train_datagen.flow_from_dataframe(
            dataframe=Shuffle_file_paths_and_labels_df,
            directory= self.data_dir,
            x_col="FilePath",
            y_col="Label",
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            subset='training') # set as training data

        self.validation_generator = self.train_datagen.flow_from_dataframe(
            dataframe=Shuffle_file_paths_and_labels_df,
            directory = self.data_dir,
            x_col="FilePath",
            y_col="Label",
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            subset='validation') # set as validation data

    def image_load_dir(self):

        self.train_generator = self.train_datagen.flow_from_directory(
            directory= self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            subset='training') # set as training data

        self.validation_generator = self.train_datagen.flow_from_directory(
            directory = self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            subset='validation') # set as validation data

    def image_check(self):
        """
        TODO: Doc string
        """
        def image_plot(generator):
            testX_sanity, testY_sanity = next(iter(generator))
            testY_sanity = testY_sanity.astype(int)

            L = 3
            W = 3

            fig, axes = plt.subplots(L,W,figsize=(12,12))
            axes = axes.ravel()

            for i in np.arange(0, L*W):
                axes[i].imshow(testX_sanity[i])
                axes[i].set_title('{}'.format(testY_sanity[i]))
                axes[i].axis('off')
            plt.subplots_adjust(hspace = 0)
            plt.show()
            plt.close()

        print("train")
        image_plot(self.train_generator)
        print("Valid")
        image_plot(self.validation_generator)