# asr_project
Automatic Speech Recognition (ASR) Project

This repository contains the implementation of the DeepSpeech2 model for the ASR task. 
To reproduce the final model's result train your model with the `src/configs/deepspeech2_augs.json` configuration for 75 epochs.

# Results
WER = $0.276992$ and CER = $$0.125911$ were achieved on `test-other` split by the final model (using beam-search + language model).

# Installation
- Clone this repository
```bash
git clone https://github.com/hzchet/asr_project.git
cd asr_project
```
- In order to install all the required packages, run the following command in your terminal:
```bash
  pip install -r requirements.txt
```
- Install language models (used for beam-search rescoring) by running the following commands
```bash
wget -P ./saved/lms/ https://www.openslr.org/resources/11/3-gram.arpa.gz
wget -P ./saved/lms/ https://www.openslr.org/resources/11/3-gram.arpa.gz
gzip -d ./saved/lms/3-gram.arpa.gz
gzip -d ./saved/lms/4-gram.arpa.gz
```
- Install the weights of the pre-trained model by running
```bash
python3 install_weights.py
```
- Copy the config into the same directory
```bash
cp src/configs/deepspeech2_augs.json saved/models/final/config.json
```

# Tests
In order to run unit tests run the following command
```bash
python3 unit_tests.py
```

# Metrics
In order to reproduce metrics on `test-clean`/`test-other` datasets, run the following command
```bash
python3 test.py -r saved/models/final/weights.pth
```
