import keras
import IPython.display
import matplotlib

import tensorflow as tf
import numpy as np

import src.helpers.print_extensions
class NetworkAnalyzer:
    """ Class to perform deep analysis of given convolutional neural network """
    def __init__(self, network_path, layer_num, img_path):
        self.network_path = network_path
        self.layer_num = layer_num
        self.img_path = img_path

    def full_analysis(self):
        """
        Function to perform full CNN analysis
        """
        self.get_filter_shapes()
        self.visualize_filters()
        self.visualize_feature_maps()

    def get_filter_shapes(self):
        """
        Function to display kernels (filters) shape in each conv2d layer of given model.e
        """
        src.helpers.print_extensions.print_subtitle("Model architecture")
        cnn_model = tf.keras.models.load_model(self.network_path)
        cnn_model.summary()
        src.helpers.print_extensions.print_subtitle("Filter shapes")
        for layer in cnn_model.layers:
            if 'conv' not in layer.name:
                continue
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)
        print("\n\n")

    def visualize_filters(self):
        """
        Function to display sample of conv filter for given layer of given network.
        """
        src.helpers.print_extensions.print_subtitle(f"Filters sample from layer {self.layer_num}")
        cnn_model = tf.keras.models.load_model(self.network_path)
        filters, biases = cnn_model.layers[self.layer_num].get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        n_filters, ix = 3, 1
        for i in range(n_filters):
            f = filters[:, :, :, i]
            for j in range(5):
                ax = matplotlib.pyplot.subplot(n_filters, 5, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                matplotlib.pyplot.imshow(f[:, :, j], cmap='gray')
                ix += 1
        matplotlib.pyplot.show()
        print("\n\n")

    def visualize_feature_maps(self):
        """
        Function to display feature maps for each layer of given network.
        Feature maps are extracted from network prediction on given image.
        """
        src.helpers.print_extensions.print_subtitle(f"Feature maps for each conv2d layer")
        cnn_model = tf.keras.models.load_model(self.network_path)
        img = keras.preprocessing.image.load_img(self.img_path, target_size=(64, 64))
        IPython.display.display(img.resize((256, 256)))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        for layer in cnn_model.layers:
            if 'conv' not in layer.name:
                continue
            model = keras.models.Model(inputs=cnn_model.inputs, outputs=layer.output)
            feature_maps = model.predict(img)
            ix = 1
            for _ in range(8):
                matplotlib.pyplot.figure(figsize=(10, 10))
                for _ in range(4):
                    ax = matplotlib.pyplot.subplot(8, 4, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    matplotlib.pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                    ix += 1
            print("\n")
            src.helpers.print_extensions.print_subtitle(f"Layer: {layer.name}")
            matplotlib.pyplot.show()
