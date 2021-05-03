#!/bin/sh

sudo apt-get update && sudo apt-get upgrade
pip install torch --no-cache-dir
pip install absl-py
pip install future
pip install h5py
pip install tqdm
pip install attrs
sudo apt install unzip


git clone https://github.com/RachanaJayaram/VizWiz-VQA.git

