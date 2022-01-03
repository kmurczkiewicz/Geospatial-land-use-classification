from setuptools import setup, find_packages

setup(
   name='geo_sat_nn_app',
   version='0.1',
   description='Geospatial land use classification tool',
   license="",
   author='Kacper Murczkiewicz',
   author_email='murczkiewiczkacper@gmail.com',
   url="https://github.com/kmurczkiewicz/Sentinel-2-Image-classification-using-CNNs",
   packages=find_packages(include=['bin', 'src'])
)
