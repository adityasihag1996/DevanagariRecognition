import os
import torch
from collections import Counter
from PIL import Image

from config import crnn_transform

def create_vocab(ground_truth_path):
    """
    Creates a vocabulary from the ground truth file.
    """
    # Read all the labels and count the characters
    char_counts = Counter()
    with open(ground_truth_path, 'r', encoding='utf-8') as file:
        for line in file:
            _, label = line.strip().split(",")
            char_counts.update(label)
    
    # Create the vocabulary by sorting characters by frequency
    vocabulary = sorted(char_counts, key=char_counts.get, reverse=True)
    
    # Add a special character for CTC blank token
    vocabulary = ['[BLANK]'] + vocabulary
    
    # Create the char-to-index mapping dictionary
    char_to_index = {char: idx for idx, char in enumerate(vocabulary)}
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    
    return char_to_index, index_to_char, vocabulary

def encode_labels(labels, char_to_index):
    """
    Encodes text labels to numerical values based on the character index mapping.
    """
    encoded_labels = [torch.LongTensor([char_to_index[char] for char in label]) for label in labels]
    return encoded_labels

def load_images_and_labels(images_folder, ground_truth_path, char_to_index):
    """
    Loads and encodes all images and labels from a given folder and ground truth file.
    """
    images = []
    labels = []

    # Read the ground truth labels
    with open(ground_truth_path, 'r', encoding='utf-8') as file:
        for line in file:
            image_file, label = line.strip().split(",")
            labels.append(label)
            img_path = os.path.join(images_folder, image_file)
            image = Image.open(img_path).convert('L')  # convert image to grayscale
            image = crnn_transform(image)
            images.append(image)

    # Encode all labels
    encoded_labels = encode_labels(labels, char_to_index)

    return images, encoded_labels

