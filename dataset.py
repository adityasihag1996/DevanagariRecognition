import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from utils import load_images_and_labels
from config import BATCH_SIZE, BLANK_TOKEN


# Custom collate_fn to handle variable-length sequences
def collate_fn(batch):
    images, targets = zip(*batch)
    
    # Pad the sequences to the maximum length in the batch
    padded_targets = pad_sequence(targets, batch_first = True, padding_value = BLANK_TOKEN)  # Assuming 0 is the padding value
    
    # No need to pad images since they are already resized to a fixed size (224, 224)
    images = torch.stack(images, dim = 0)
    
    return images, padded_targets

class PreEncodedDataset(Dataset):
    def __init__(self, encoded_images, encoded_labels):
        self.encoded_images = encoded_images
        self.encoded_labels = encoded_labels
        
    def __len__(self):
        return len(self.encoded_images)

    def __getitem__(self, idx):
        return self.encoded_images[idx], self.encoded_labels[idx]
    

def create_dataset_and_dataloader(images_path, ground_truth_path, vocab_path, test_split = 0.2):
    with open(vocab_path, 'rb') as f:
        char_to_index = pickle.load(f)

    index_to_char = {idx: char for char, idx in char_to_index.items()}

    # Load and encode all images and labels
    encoded_images, encoded_labels = load_images_and_labels(images_path, ground_truth_path, char_to_index)

    full_dataset = PreEncodedDataset(encoded_images, encoded_labels)
    
    # Split the dataset into training and validation sets
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create training and validation data loaders
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = collate_fn)
    
    return train_loader, test_loader
