import matplotlib.pyplot as plt
import numpy as np

def train_plot(model_history,NUM_EPOCHS):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), model_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), model_history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), model_history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCHS), model_history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()