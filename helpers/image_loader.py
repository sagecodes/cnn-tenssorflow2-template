from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class train_generator:
    """
    TODO: Add doc string
    """

    def __init__(self,img_height,img_width,BATCH_SIZE, data_dir, df):
        # Shuffle dataFrame 
        Shuffle_file_paths_and_labels_df = df.sample(frac=1).reset_index(drop=True)
        # Create Data Generators

        train_datagen = ImageDataGenerator(
            rescale=1./255,
        #     shear_range=0.2,
        #     zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2) # set validation split

        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=Shuffle_file_paths_and_labels_df,
            directory= data_dir,
            x_col="FilePath",
            y_col="Label",
            target_size=(img_height, img_width),
            batch_size=BATCH_SIZE,
            seed=42,
            shuffle=True,
            class_mode='categorical',
            subset='training') # set as training data

        self.validation_generator = train_datagen.flow_from_dataframe(
            dataframe=Shuffle_file_paths_and_labels_df,
            directory = data_dir,
            x_col="FilePath",
            y_col="Label",
            target_size=(img_height, img_width),
            batch_size=BATCH_SIZE,
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