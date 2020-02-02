

def train(self,BATCH_SIZE,NUM_EPOCHS, train_data,val_data):

        self.train_history = self.model.fit_generator(
        train_data,
        steps_per_epoch = train_data.samples // BATCH_SIZE,
        validation_data = val_data, 
        validation_steps = val_data.samples // BATCH_SIZE,
        epochs = NUM_EPOCHS, 
        verbose=1)

        return self.train_history


def plot_train_history(self,NUM_EPOCHS):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, NUM_EPOCHS),
                    self.train_history.history["loss"],
                    label="train_loss")
        plt.plot(np.arange(0, NUM_EPOCHS),
                    self.train_history.history["val_loss"],
                    label="val_loss")
        plt.plot(np.arange(0, NUM_EPOCHS),
                    self.train_history.history["accuracy"],
                    label="train_acc")
        plt.plot(np.arange(0, NUM_EPOCHS),
                    self.train_history.history["val_accuracy"],
                    label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()
        plt.close()


def save_model():
    pass

def load_model():
    pass

def predict():
    pass

def test_model():
    pass