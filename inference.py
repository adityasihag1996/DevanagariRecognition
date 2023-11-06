import torch
from torchvision import transforms
from torch.nn.functional import log_softmax
from itertools import groupby
from PIL import Image
import argparse
import pickle

from model import CRNN
from config import crnn_transform, DEVICE, NUM_CLASSES, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, BIDIRECTIONAL


def decode_predictions(preds, charset):
    """
    Decode the raw predictions of the network into strings.
    """
    decoded_preds = []
    pred_values = preds.max(2)[1]  # Get the max index from the softmax output
    for i in range(pred_values.size(1)):
        pred = pred_values[:, i].tolist()
        # CTC decoding
        # Collapse repeated characters and remove the CTC blank token (assumed to be 0)
        pred = [p for p, _ in groupby(pred)]
        pred_str = ''.join([charset[p] for p in pred if p != 0])
        decoded_preds.append(pred_str)
    return decoded_preds


def infer(model, image_path, vocab):
    """
    Run inference on an input image and return the predicted string.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # grayscale
    image = crnn_transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and send to device
    
    # Set the model to evaluation mode
    model.eval()
    
    # Forward pass to get the raw predictions
    with torch.no_grad():
        raw_preds = model(image)
        # Permute so the time dimension is first for CTC decoding
        raw_preds = raw_preds.permute(1, 0, 2)
        # Apply log_softmax to get log probabilities
        log_probs = log_softmax(raw_preds, dim=2)
    
    # Decode the raw predictions to a string
    decoded_preds = decode_predictions(log_probs, vocab)[0]
    
    return decoded_preds


def parse_opt():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset of images with text.")

    parser.add_argument("-mp", "--model_path", type=str, required=True,
                    help="Path to the model state-dict.")
    parser.add_argument("-ip", "--image_path", type=str, required=True,
                    help="Path to the image.")
    parser.add_argument("-vp", "--vocab_path", type=str, required=True,
                    help="Path to the vocab pickle file.")

    return parser.parse_args()



if __name__ == '__main__':
    # args
    args = parse_opt()

    model_path = args.model_path
    image_path = args.image_path
    vocab_path = args.vocab_path

    with open(vocab_path, 'rb') as f:
        char_to_index = pickle.load(f)

    model = CRNN(NUM_CLASSES, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, BIDIRECTIONAL).to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    pred = infer(model, image_path, char_to_index)
    print(pred)
    
    