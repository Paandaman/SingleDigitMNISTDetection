# SingleDigitMNISTDetection
A classifier trained on a single sample per class from the MNIST dataset.

Trained using the approach described in the paper 'Unsupervised Data Augmentation for Consistency Training" in order to utilize the 'unlabeled' samples. Evaluated on a 1k subset of the MNIST test dataset (100 samples per digit).

### Requirements
```
python3
numpy
Pillow
pyyaml
tensorboard
torch
torchvision
```
### Installation

```bash
git clone https://github.com/Paandaman/SingleDigitMNISTDetection.git
cd SingleDigitMNISTDetection
#Install required packages
conda install --file requirements.txt
```
### Train
```
python main.py
```

### Keep track of training progress
```
cd SingleDigitMNISTDetection
python -m tensorboard.main --logdir=.
```

Various parameters can be changed in the config.yaml file.
