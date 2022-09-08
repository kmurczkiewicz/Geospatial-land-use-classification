# Geospatial-land-use-classification

- **Documentation**: https://kmurczkiewicz.github.io/Docs-Geospatial-land-use-classification/ 
- **Dataset**: https://www.kaggle.com/apollo2506/eurosat-dataset 


### **Repository content** 

- Python application to automate designing CNN solutions for geospatial land use classification problem.
- Module to perform land use map generation for given satellite image
- Automated documentation generation using Sphinx: https://github.com/kmurczkiewicz/Docs-Geospatial-land-use-classification

### **Features**

- Dataset analysis, data preparation and management
- Executor based approach
- MLP/CNN designing for classification tasks, training and testing on given dataset
- Hyper-parameters tuning and optimization
- Land use map generation with given network model on given satellite image
- Network storing, testing and analysis

## Installation (Windows OS)

```shell
# Clone the repository
git clone https://github.com/kmurczkiewicz/Geospatial-land-use-classification.git
cd Geospatial-land-use-classification

# Create and activate new python virtual environment
python -m virtualenv geo_env
geo_env\Scripts\activate

# Install the app and dependencies
python -m pip install -r requirements.txt
python -m pip install -e .

# Initialize local working environment
init.bat
```

## Usage
Clone the EuroSAT dataset from Kaggle to: **Geospatial-land-use-classification\artefacts\dataset** \
Only EuroSAT dataset is required, **EuroSATallBands part** (which size is much greater) is not required.

```shell
# Run jupyter notebook
jupyter notebook
```

### 1. Train the model
Run notebook: **notebooks/nn_full_flow.ipynb** \
Adjust the OPTIMIZERS list and ACTIVATIONS lists. For single model generation define mentioned lists size to 1, for instance:
```python
OPTIMIZERS = [{"optimizer" : tf.keras.optimizers.Adamax}]
ACTIVATIONS = ['relu']
```
Define number of epochs, batch size, loss_function (from tf.keras.losses), metrics, if model should be saved and the model **architecture**. \
Available architectures are located in: **Geospatial-land-use-classification/src/nn_library/nn_architectures.py** \
<br/>
User can edit those architectures or created new ones. Each architecture should be a seperate function, once new architecture is created, it has to be added to **NN_ARCHITECTURES** dictonary in **Geospatial-land-use-classification/src/execution/base_executor.py**
<br/><br/>
Output models are saved in **Geospatial-land-use-classification\artefacts\models_pb**

### 2. Generate land use map
Get satellite image and add it to **Geospatial-land-use-classification\artefacts\sat_images**. <br/>
For instance: **Geospatial-land-use-classification\artefacts\sat_images\new_york.jpg**
<br/><br/>
Example source of satellite images could be **Google Earth Pro** desktop application, which allows to export satellite images in .jpg format in max resolution of 4800x2869.<br/>
https://www.google.com/earth/versions/#earth-pro
<br/><br/>
Run notebook: **notebooks/nn_land_classfication_use_case.ipynb** \
Adjust the parameters:
- **network_name** - string name of the model to be used from **Geospatial-land-use-classification\artefacts\models_pb** <br/>
For instance: **"dir_1/network_D_1736080922"**
- **sat_img_list** - list of satellite images from **Geospatial-land-use-classification\artefacts\sat_images** <br/>
For instance: **["new_york.jpg"]**
<br/><br/>
### 3. Other features
- **notebooks/data_analysis.ipynb** - quantitative analysis of EuroSAT dataset
- **notebooks/nn_analysis.ipynb** - analyze saved models details and details of specified layers
- **notebooks/nn_hyper_parameters_tuning.ipynb** - tune hyper-parameters using Keras Tuner: https://keras.io/keras_tuner/
- **notebooks/nn_testing.ipynb** - test models for precision, recall and F1-Score
<br/><br/>
## Example geospatial land use classification for satellite imagery

### 1. Raw satellite image
![elblag](https://user-images.githubusercontent.com/71273151/180648900-4fc98760-70ba-45ac-9fbd-7b99d18651dd.png)
**source**: Google Earth Pro 7.3, (2017) Elblag 54°09'21.51"N, 19°24'16.16"E, elevation 15.00 km

### 2. Land use mask generated using CNN
![mask](https://user-images.githubusercontent.com/71273151/180648915-7c69834c-4319-4837-b07c-c533c124d300.png)

### 3. Satellite image with land use mask applied
![mask_applied](https://user-images.githubusercontent.com/71273151/180648919-de392b9d-1097-45c5-b478-6ab47c05c6fa.png)


### 4. Land use distribution
![diagram](https://user-images.githubusercontent.com/71273151/180648924-b010817c-38a3-4111-bf6e-8a0e652207ff.png)
