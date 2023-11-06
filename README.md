# Convolutional Recurrent Neural Network (CRNN) for Image-Based Sequence Recognition (Devanagari Recognition)

This project implements a Convolutional Recurrent Neural Network (CRNN) in PyTorch for the task of image-based sequence recognition. This model combines CNN with RNN to process images which include sequences of items, such as text in various fonts and handwriting, or objects in a series. The primary use case demonstrated in this project is optical character recognition (OCR).

## Project Overview

The CRNN model in this project has been designed to read sequences of characters from images. It uses a modified convolutional neural network for feature extraction, and a recurrent neural network to sequence these features. The model outputs a string of text as interpreted from the input image.

## Features

- Configurable LSTM for sequence modeling
- CTC (Connectionist Temporal Classification) loss for sequence prediction
- Training and inference scripts

## Requirements

To install the required libraries, run the following command:

```
pip install -r requirements.txt
```

## Installation
Clone the repository to your local machine:

```
git clone https://github.com/adityasihag1996/DevanagariRecognition.git
cd DevanagariRecognition
```

## Usage

**_NOTE:-_** Before running the scripts, please adjust the paths accordingly in `config.py`.
**_A sample checkpoint has been provided in the repo for testing, `resnet18_lstm_crnn.pth`_**

To train the model, run:

```
python train.py
```

For inference on a single image:

```
python inference.py --image_path /path/to/your/image.jpg --model_path /path/to/your/model.pth --vocab_path /path/to/your/vocab.pickle
```

## Sample Image
Below is a sample image, and model prediction:-

![Sample Image](/sample_dev.png "Sample Image Title")
कड़ाईमुजुक्कू

## To-Do

- [ ] Customizable CNN Backbone.
- [ ] Evaluation Scripts.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.
