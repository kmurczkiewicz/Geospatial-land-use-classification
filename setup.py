from setuptools import setup, find_packages

setup(
   name='cnn_dev_app',
   version='0.1',
   description='CNN satellite image classification module',
   license="",
   author='Kacper Murczkiewicz',
   author_email='murczkiewiczkacper@gmail.com',
   url="https://github.com/kmurczkiewicz/Sentinel-2-Image-classification-using-CNNs",
   packages=find_packages(include=['bin', 'src'])
)
