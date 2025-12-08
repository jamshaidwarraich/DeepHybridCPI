import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from utils import set_seed

set_seed(42)

# --------------------- Custom Piecewise Activation ---------------------
class CustomPiecewiseActivation(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, x, torch.sin(x))

# ------------------------ Protein Encoder ------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinCNNLSTM(nn.Module):
    def __init__(self, protein_dict_len=25, embedding_size=128, num_filters=32, protein_filter_lengths=8):
        super(ProteinCNNLSTM, self).__init__()
        
        # Protein embedding layer
        self.embedding = nn.Embedding(num_embeddings=protein_dict_len+1, 
                                     embedding_dim=embedding_size,
                                     padding_idx=0)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=embedding_size, 
                              out_channels=num_filters, 
                              kernel_size=protein_filter_lengths, 
                              padding='valid')
        
        self.conv2 = nn.Conv1d(in_channels=num_filters, 
                              out_channels=num_filters*2, 
                              kernel_size=protein_filter_lengths, 
                              padding='valid')
        
        self.conv3 = nn.Conv1d(in_channels=num_filters*2, 
                              out_channels=num_filters*3, 
                              kernel_size=protein_filter_lengths, 
                              padding='valid')
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_filters*3,
                           hidden_size=num_filters*3,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
    
        self.custom_activation = CustomPiecewiseActivation()
        self.dropout = nn.Dropout(0.2)

        
        # Final linear layer to reduce dimension to num_filters * 3
        self.linear = nn.Linear(num_filters * 3 * 2, num_filters * 3)  # Input is conv+lstm features

    def forward(self, x):
        # Input shape: (batch_size, protein_max_len)
        
        # Embedding layer
        x = self.embedding(x)  # shape: (batch_size, protein_max_len, embedding_size)
        
        # Prepare for Conv1d (needs channels first)
        x = x.permute(0, 2, 1)  # shape: (batch_size, embedding_size, protein_max_len)
        
        # Convolutional layers with custom activation
        conv1_out = self.custom_activation(self.conv1(x))
        conv1_out = self.dropout(conv1_out)

        conv2_out = self.custom_activation(self.conv2(conv1_out))
        conv2_out = self.dropout(conv2_out)

        conv3_out = self.custom_activation(self.conv3(conv2_out))

        
        # Global max pooling for conv features
        conv_features = F.max_pool1d(conv3_out, kernel_size=conv3_out.shape[2]).squeeze(2)
        
        # Prepare for LSTM (needs channels last again)
        lstm_input = conv3_out.permute(0, 2, 1)  # shape: (batch_size, seq_len, num_filters*3)
        
        # LSTM layer
        lstm_out, _ = self.lstm(lstm_input)
        
        # Global max pooling for LSTM features
        lstm_features = torch.max(lstm_out, dim=1)[0]
        
        # Combine features
        combined_features = torch.cat([conv_features, lstm_features], dim=1)
        
        # Apply final linear layer
        protein_features = self.linear(combined_features)
        
        return protein_features

# ------------------------ Graph Components ------------------------
class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError(f'expected 2D input (got {input.dim()}D input)')

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                exponential_average_factor = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)
        self.activation = CustomPiecewiseActivation()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        data.x = self.activation(self.norm(self.conv(x, edge_index)))
        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, growth_rate * bn_size)
        self.conv2 = GraphConvBn(growth_rate * bn_size, growth_rate)

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]
        concated = torch.cat(data.x, dim=1)
        data.x = concated

        data = self.conv1(data)
        data = self.conv2(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size
            )
            self.add_module(f'denselayer{i+1}', layer)

    def forward(self, data):
        features = [data.x]
        for layer in self.values():
            data = layer(data)
            features.append(data.x)
            data.x = features
        data.x = torch.cat(features, dim=1)
        return data

class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(8, 8, 8), bn_sizes=(2, 2, 2)):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', GraphConvBn(num_input_features, 32))
        ]))
        num_features = 32

        for i, (num_layers, bn_size) in enumerate(zip(block_config, bn_sizes)):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate

            trans = GraphConvBn(num_features, num_features // 2)
            self.features.add_module(f'transition{i+1}', trans)
            num_features = num_features // 2

        self.classifier = nn.Linear(num_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = global_mean_pool(data.x, data.batch)
        x = self.classifier(x)
        return x

# ------------------------ Final Model ------------------------
class MGraphCPI(nn.Module):

    def __init__(self, embedding_size=128, num_filters=32, out_dim=2):
        super().__init__()


        self.protein_encoder = ProteinCNNLSTM(
            protein_dict_len=25,  # Number of unique amino acids
            embedding_size=embedding_size,
            num_filters=num_filters,
            protein_filter_lengths=8  # Or your preferred kernel size
        )

        self.ligand_encoder = GraphDenseNet(
            num_input_features=87,
            out_dim=num_filters * 3,
            block_config=[8, 8, 8],
            bn_sizes=[2, 2, 2]
        )

        self.classifier = nn.Sequential(
            nn.Linear(96 + num_filters * 3, 1024),
            CustomPiecewiseActivation(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            CustomPiecewiseActivation(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            CustomPiecewiseActivation(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )

    def forward(self, data, return_embedding=False):
        protein_x = self.protein_encoder(data.target)
        ligand_x = self.ligand_encoder(data)
        x = torch.cat([protein_x, ligand_x], dim=1)
        x = self.classifier(x)
        return x
