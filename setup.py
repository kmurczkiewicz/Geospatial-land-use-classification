from setuptools import setup, find_namespace_packages

setup(
   name='cnn_dev_app',
   version='0.1',
   description='CNN satellite image classification module',
   license="",
   author='Kacper Murczkiewicz',
   author_email='murczkiewiczkacper@gmail.com',
   url="https://github.com/kmurczkiewicz/Sentinel-2-Image-classification-using-CNNs",
   packages=[
       'src.data',
       'src.executor',
       'src.helpers',
       'src.nn_operations',
    ]
   #install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
   #scripts=[
   #         'scripts/cool',
    #        'scripts/skype',
     #      ]
)