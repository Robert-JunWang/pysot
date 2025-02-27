# Installation

This document contains detailed instructions for installing dependencies for PySOT. We recommand using the [install.sh](install.sh). The code is tested on an Ubuntu 16.04 system with Nvidia GPU (We recommand 1080TI / TITAN XP).

### Requirments
* Python 3.7.
* Nvidia GPU.
* PyTorch 0.4.1
* yacs
* pyyaml
* matplotlib
* tqdm
* OpenCV

## Step-by-step instructions

####  Compile Python3.7

https://tecadmin.net/install-python-3-7-on-ubuntu-linuxmint/

####  Create a virtual environment

```
virtualenv --system-site-packages -p python3.7 $HOME/venv37
source $HOME/venv37/bin/activate


```

#### Install numpy/pytorch/opencv
```
pip install numpy
pip install torch==0.4.1.post2
pip install opencv-python
```

#### Install other requirements
```
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

```

#### Build extensions
```
python setup.py build_ext --inplace
```

