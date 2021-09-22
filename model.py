import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:,:-1] 
        embeds = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_out, hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []
        hidden = (torch.randn(1, 1, self.hidden_size).to(inputs.device),
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(outputs)
            outputs = outputs.squeeze(1)
            target_index = outputs.argmax(dim=1)
            captions.append(target_index.item())
            inputs = self.word_embeddings(target_index.unsqueeze(0))
        return captions