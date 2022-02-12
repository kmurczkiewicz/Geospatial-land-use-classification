import os
import PIL
import pathlib
import re
import IPython.display
import matplotlib.pyplot
import natsort

import numpy as np
import tensorflow as tf

import src.helpers.print_extensions


class SatelliteImageClassifier:
    """
    Class defining example client use case for land use classification CNN model.
    """
    def __init__(self, paths, network_name, sat_images: list):
        self.PATHS = paths
        self.network_name = network_name
        self.sat_images = sat_images
        self.CLASSES = {
            "AnnualCrop": {"color" : "#FFAE00", "mapped_amount" : 0},
            "Forest": {"color" : "#1EFF1E", "mapped_amount" : 0},
            "HerbaceousVegetation": {"color" : "#79E0A8", "mapped_amount" : 0},
            "Highway": {"color" : "#FF1EF1", "mapped_amount" : 0},
            "Industrial": {"color" : "#F50318", "mapped_amount" : 0},
            "Pasture": {"color" : "#A2F91D", "mapped_amount" : 0},
            "PermanentCrop": {"color" : "#DFD433", "mapped_amount" : 0},
            "Residential": {"color" : "#98A0A2", "mapped_amount" : 0},
            "River": {"color" : "#06BDF8", "mapped_amount" : 0},
            "SeaLake": {"color" : "#0648F8", "mapped_amount" : 0},
        }

    def _load_sat_image_as_array(self, sat_image):
        """
        Function to load a satellite image and resize its width and height to multiplication of 64.

        :param sat_image: str name of satellite image to be loaded as numpy array.
        :return: numpy array representing given satellite image
        """
        img = PIL.Image.open(os.path.join(self.PATHS["SAT_IMG_PATH"], sat_image))
        original_width, original_height = img.size
        fixed_width = original_width - (original_width % 64)
        fixed_height = original_height - (original_height % 64)
        img = img.resize((fixed_width, fixed_height), PIL.Image.ANTIALIAS)

        src.helpers.print_extensions.print_subtitle(f"1. Original satellite image - {sat_image}")
        IPython.display.display(img)

        return np.asarray(img)

    def _split_sat_image_to_tiles(self, sat_image):
        """
        Function to split satellite image into multiple 64x64 tiles, which are saved locally.

        :param sat_image: str name of satellite image to be splitted into 64x64 tiles.
        :return: dict containing image dimensions in number of tiles
        """
        tiles = [sat_image[x:x + 64, y:y + 64] for x in range(0, sat_image.shape[0], 64) for y in
                 range(0, sat_image.shape[1], 64)]
        width, height = PIL.Image.fromarray(sat_image).size

        print(f"Width: {int(width / 64)} tiles of shape 64x64")
        print(f"Height: {int(height / 64)} tiles of shape 64x64")
        print(f"Image was splitted into: {int(width / 64) * int(height / 64)} tiles of shape 64x64")

        index = 0
        for tile in tiles:
            img = PIL.Image.fromarray(tile, 'RGB')
            img.save(os.path.join(self.PATHS["SAT_TILES_PATH"], f"{index}.png"))
            index += 1

        return {
            "tiles_in_row": int(width / 64),
            "tiles_in_col": int(height / 64)
        }

    def _predict_land_use(self):
        """
        Function to perform classification on each tile present in default tiles directory. Given CNN
        model is loaded to perform inference on each tile. Depending on inference result, new tile with
        class color representing inference result is saved locally for each source tile.
        """
        network = src.nn_library.network.Neural_network()
        network.model = tf.keras.models.load_model(os.path.join(self.PATHS["NETWORK_SAVE_DIR"], self.network_name))

        tile_list = os.listdir(self.PATHS["SAT_TILES_PATH"])
        tile_list = natsort.natsorted(tile_list)

        for tile in tile_list:
            tile_img = PIL.Image.open(os.path.join(self.PATHS["SAT_TILES_PATH"], tile))
            predicted_class = network.single_class_prediction(
                np.array(np.asarray(tile_img) / 255.0).reshape(1, 64, 64, 3)
            )
            predicted_label = list(self.PATHS["LABEL_MAP"].keys())[
                list(self.PATHS["LABEL_MAP"].values()).index(predicted_class)
            ]
            single_mapped_tile = PIL.Image.new(
                "RGB", (64, 64), PIL.ImageColor.getrgb(self.CLASSES[predicted_label]["color"])
            )
            self.CLASSES[predicted_label]["mapped_amount"] += 1
            single_mapped_tile.save(os.path.join(self.PATHS["SAT_MAP_TILES_PATH"], pathlib.Path(tile).stem + ".png"))

    def _generate_land_use_map(self, tiles_row_col, sat_img_name, original_img_arr):
        """
        Function to generate land use map for given satellite image. Land is map is generated based on
        mapped tiles (tiles which color represents class). First, image rows are generated from tiles, later
        on rows are concatenated into single image, which is final land use classification map.

        :param tiles_row_col: dict containing image dimensions in number of tiles
        :param sat_image: str name of satellite image for which land use map shall be generated
        :param original_img_arr: numpy array representing original sat image
        """
        mapped_tile_list = os.listdir(self.PATHS["SAT_MAP_TILES_PATH"])
        mapped_tile_list = natsort.natsorted(mapped_tile_list)
        single_row = PIL.Image.new('RGBA', (tiles_row_col["tiles_in_row"] * 64, 64))

        tile_num = 0
        mapped_rows = []

        for tile in mapped_tile_list:
            if tile_num < tiles_row_col["tiles_in_row"]  - 1:
                single_row.paste(
                    PIL.Image.open(os.path.join(self.PATHS["SAT_MAP_TILES_PATH"], tile)),
                    (tile_num * 64, 0)
                )
                tile_num += 1
                continue
            single_row.paste(
                PIL.Image.open(os.path.join(self.PATHS["SAT_MAP_TILES_PATH"], tile)),
                (tile_num * 64, 0)
            )
            mapped_rows.append(single_row)
            tile_num = 0
            single_row = PIL.Image.new('RGBA', (tiles_row_col["tiles_in_row"] * 64, 64))

        mapped_image = PIL.Image.new(
            'RGBA',
            (
                tiles_row_col["tiles_in_row"] * 64,
                tiles_row_col["tiles_in_col"] * 64
            )
        )

        row_num = 0
        for row in mapped_rows:
            mapped_image.paste(
                row,
                (0, row_num * 64)
            )
            row_num += 1

        original_sat_img = PIL.Image.fromarray(original_img_arr)

        # Set mask transparency
        mapped_image.putalpha(80)

        original_sat_img.paste(mapped_image, (0, 0), mapped_image)

        print("\n")
        src.helpers.print_extensions.print_subtitle(f"2. Land use classification mask - {sat_img_name}")
        IPython.display.display(mapped_image)
        print("\n")
        src.helpers.print_extensions.print_subtitle(f"3. Satellite image with applied land use mask - {sat_img_name}")
        IPython.display.display(original_sat_img)

    def _reset_mapped_classes(self):
        """
        Function to reset mapped classes.
        """
        print("\n\n\n")
        for key, value in self.CLASSES.items():
            value["mapped_amount"] = 0

    def _clear_tile_dirs(self):
        """
        Function to remove all tiles and mapped tiles, as for multiple sat images
        empty tiles dirs are required.
        """
        for tile in os.listdir(self.PATHS["SAT_TILES_PATH"]):
            os.remove(os.path.join(self.PATHS["SAT_TILES_PATH"], tile))

        for mapped_tile in os.listdir(self.PATHS["SAT_MAP_TILES_PATH"]):
            os.remove(os.path.join(self.PATHS["SAT_MAP_TILES_PATH"], mapped_tile))

    def _set_bar_text_value(self, bar_plot, bar_value):
        """
        Function to display bar value above bar plot.

        :param bar_plot: matplotlib.pyplot.bar object
        :param bar_value: value to be displayed above bar
        """
        height = bar_plot.get_height()
        matplotlib.pyplot.text(
            bar_plot.get_x() + bar_plot.get_width() / 2., 1.02 * height, bar_value, ha='center', va='bottom', rotation=0
        )

    def _plot_class_distribution(self, sat_img):
        """
        Function to plot predicted classes distribution on given satellite image.

        :param sat_image: str name of satellite image
        """
        print("\n")
        src.helpers.print_extensions.print_subtitle(f"4. Predicted class distribution - {sat_img}")
        x = [key for key, value in self.CLASSES.items() if value["mapped_amount"] > 0]
        y = [value["mapped_amount"] for value in self.CLASSES.values() if value["mapped_amount"] > 0]
        matplotlib.pyplot.figure(figsize=(16, 7))
        bar_list = matplotlib.pyplot.bar(x, y)
        matplotlib.pyplot.ylim(ymax=max(y)+(max(y)/10), ymin=0)
        for index, class_name in enumerate(x):
            self._set_bar_text_value(bar_list[index], y[index])
            bar_list[index].set_color(self.CLASSES[class_name]["color"])
        matplotlib.pyplot.xlabel('Class')
        matplotlib.pyplot.ylabel(f'Amount on {sat_img}')
        matplotlib.pyplot.show()

    def run_classification(self):
        """
        Function to run classification process on given sat images. For each image, land use mask is generated
        using provided CNN model. Further, land use mask is applied to source image.
        """
        for sat_img in self.sat_images:
            self._clear_tile_dirs()
            sat_img_array = self._load_sat_image_as_array(sat_img)
            tiles_row_col = self._split_sat_image_to_tiles(sat_img_array)
            self._predict_land_use()
            self._generate_land_use_map(tiles_row_col, sat_img, sat_img_array)
            self._plot_class_distribution(sat_img)
            self._reset_mapped_classes()
