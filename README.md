# Training Machine Learning models with galaxy image data -- using supervised and self-supervised learning

## Dataset
We are using GalaxyML dataset which is available in Zenodo: https://zenodo.org/records/13878122. We have used the 64x64 size images. The scripts assume that the data `(*.hdf5)` files are downloaded in a directory named ```Data/```. If your data is located in a different location please change the file paths in `prepare_data.py` accordingly.

## Supervised Learning
For supervised learning two kinds of architectures are used: ```ResNetRegression()``` (based on convolutional neural network) and ```ViTRegression()``` (based on transformer). The codes for these architectures are included in ```supervised_module.py```. The same script also contains a class ```Supervised()``` for training the mentioned architectures in a supervised manner using ```MSELoss``` of ```PyTorch```.
