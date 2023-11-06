import torch.nn as nn
import torch.nn.init as init
from torchvision import models

# CRNN model
class CRNN(nn.Module):
    def __init__(self, num_classes = 37, rnn_hidden_size = 256, rnn_num_layers = 1, bidirectional = True):
        super(CRNN, self).__init__()
        
        # Initialize the ResNet, remove the last fully connected layer and the average pooling layer
        self.cnn = models.resnet18(pretrained = True)
        self.cnn = nn.Sequential(
            # Keep initial layers intact (conv1, bn1, relu, maxpool)
            *list(self.cnn.children())[:4],
            # Consider keeping the first block intact to preserve some depth
            self.cnn.layer1,
            # Reduce the other blocks
            *list(self.cnn.layer2.children())[:-2],
            *list(self.cnn.layer3.children())[:-2],
            *list(self.cnn.layer4.children())[:-2],
            # Do not use the original avgpool and fc layers
        )
        # Grayscale images have 1 channel
        self.cnn[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.rnn_input_size = 64 * 56
        
        # LSTM as the sequence model
        self.lstm = nn.LSTM(self.rnn_input_size, rnn_hidden_size, rnn_num_layers, batch_first=True, bidirectional = bidirectional)
        
        # Fully connected layer for character classification
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # Times 2 for bidirectional

        for name, param in self.lstm.named_parameters():
          if 'weight_ih' in name:
              init.xavier_uniform_(param.data)
          elif 'weight_hh' in name:
              init.xavier_uniform_(param.data)
          elif 'bias' in name:
              # Biases can be set to zero or using a constant value
              param.data.fill_(0)

    def forward(self, x):
        cnn_out = self.cnn(x)

        # Prepare data for the LSTM
        batch, channels, height, width = cnn_out.size()
        cnn_out = cnn_out.view(batch, channels * height, width)  # Combine channels and height
        cnn_out = cnn_out.permute(0, 2, 1)  # Reshape to (batch, width, channels * height)
        
        # Forward pass through LSTM
        recurrent, _ = self.lstm(cnn_out)
        
        # Forward pass through fully connected layer for classification
        out = self.fc(recurrent)
        
        # Softmax is not explicitly applied because it's included in the CTCLoss during training
        return out

