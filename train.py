import torch
import torch.optim as optim
from torch.nn import CTCLoss

from model import CRNN
from dataset import create_dataset_and_dataloader
from config import NUM_EPOCHS, LR, DEVICE, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, BIDIRECTIONAL, NUM_CLASSES, BLANK_TOKEN
from config import images_path, ground_truth_path, vocab_path


def runner(model, train_loader, num_epochs, learning_rate, device):
    ctc_loss = CTCLoss(blank = BLANK_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    model.train()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Permute outputs to have time step first which is required for CTC loss
            outputs = outputs.permute(1, 0, 2)  # (T, N, C)

            input_lengths = torch.full(
                size=(outputs.size(1),),  # batch size
                fill_value=outputs.size(0),  # time steps
                dtype=torch.long
            ).to(device)
            
            target_lengths = torch.LongTensor([len(t) for t in targets]).to(device)
            
            # Ensure targets is a flat 1D tensor
            targets = torch.cat([t for t in targets])  # Concatenate all target tensors
            
            loss = ctc_loss(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        torch.save(model.state_dict(), f'resnet18_lstm_crnn_e{epoch}.pth')

    print('Finished Training!')



if __name__ == '__main__':
    model = CRNN(NUM_CLASSES, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, BIDIRECTIONAL).to(DEVICE)

    train_loader, test_loader = create_dataset_and_dataloader(
        images_path = images_path,
        ground_truth_path = ground_truth_path,
        vocab_path = vocab_path,
        test_split = 0.2
    )

    runner(model, train_loader, NUM_EPOCHS, LR, DEVICE)


