from torchvision import transforms
import torch

crnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Single channel normalization
])

NUM_EPOCHS = 150
LR = 0.00025
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLANK_TOKEN = 0
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 1
BIDIRECTIONAL = True
NUM_CLASSES = 75  # including blank token

images_path = "gen_single_images/"
ground_truth_path = "ground_truths_single.txt"
vocab_path = "vocab_devanagri.pickle"
