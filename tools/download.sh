#!/bin/sh

sudo apt-get update && sudo apt-get upgrade
pip install torch --no-cache-dir
pip install absl-py
pip install future
pip install h5py
pip install tqdm
pip install attrs
sudo apt install unzip


git clone https://github.com/RachanaJayaram/Cross-Attention-VizWiz-VQA
cd Cross-Attention-VizWiz-VQA

# Get the preprocessed data from here:
# https://drive.google.com/file/d/1s7C2ek_mSKgmR_Tk_cjHuULHEo4gUsxs/view?usp=sharing

# Ref: https://qr.ae/pG6hRe
# Get an OAuth token:
# Go to OAuth 2.0 Playground(https://developers.google.com/oauthplayground/)
# In the “Select the Scope” box, scroll down, expand “Drive API v3”, and select https://www.googleapis.com/auth/drive.readonly
# Click “Authorize APIs” and then “Exchange authorization code for tokens”. Copy the “Access token”; you will be needing it below.

curl -H "Authorization: Bearer <your_auth_token>" https://www.googleapis.com/drive/v3/files/1s7C2ek_mSKgmR_Tk_cjHuULHEo4gUsxs?alt=media -o data.zip
unzip data.zip 