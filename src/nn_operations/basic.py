"""Imports"""
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot

# Custom Imports
import src.helpers.print_extensions
import src.helpers.timer


""" Initialize, train and save model functions"""
def create_cnn_model():
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu'))
    
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(64, activation='relu'))
    cnn_model.add(layers.Dense(10))
    src.helpers.print_extensions.print_subtitle("Model summary")
    cnn_model.summary()
    
    return cnn_model


def compile_cnn_model(
    cnn_model,
    optimizer='adam',#tf.keras.optimizers.Adam(), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
):
    cnn_model.compile(optimizer, loss, metrics)

 
def train_cnn_model(cnn_model, data):
    train_history = cnn_model.fit(
        data["X_train"],
        data["y_train"],
        epochs=10,
        validation_data=(data["X_val"], data["y_val"])
    )
    return train_history


def test_cnn_model(cnn_model, data):
    test_loss, test_acc = cnn_model.evaluate(data["X_test"],  data["y_test"], verbose=2)
    print(f"Test accuracy: {test_acc}")


def plot_model_result(training_history):
    matplotlib.pyplot.plot(training_history.history['accuracy'], label='accuracy')
    matplotlib.pyplot.plot(training_history.history['val_accuracy'], label = 'val_accuracy')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Accuracy')
    matplotlib.pyplot.ylim([0.5, 1])
    matplotlib.pyplot.legend(loc='lower right')
    

def save_model(cnn_model, name, directory):
    date_str = datetime.datetime.now().strftime("%H%M%d%m%y")
    # https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#freeze-the-tensorflow-model
    tf.saved_model.save(cnn_model, f"{directory}\\{name}_{date_str}")