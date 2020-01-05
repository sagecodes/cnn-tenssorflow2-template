from tensorflow.keras.preprocessing.image import ImageDataGenerator



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