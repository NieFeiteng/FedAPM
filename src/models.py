from torch import nn
import torch.nn.functional as F
import torch
import math

import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from typing import Dict, Iterable, Optional

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size1=128, hidden_size2=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

        

    def load_first_n_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[:n]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)



    def load_remaining_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[n:]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)

        


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def load_first_n_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[:n]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)

    def load_remaining_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[n:]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)




    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1) 
        
        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def set_trainable_layers(self, n, train_first_n=True):
        """
        Set which layers are trainable based on the number of layers `n`.
        If `train_first_n` is True, train the first `n` layers and freeze the rest.
        If `train_first_n` is False, freeze the first `n` layers and train the rest.
        """
        all_layers = [
            self.conv1, self.bn1, 
            *self.layer1, *self.layer2, *self.layer3, *self.layer4,
            self.fc
        ]
        
        # Flatten all layers into a list of parameters
        layer_params = []
        for layer in all_layers:
            for param in layer.parameters():
                layer_params.append(param)

        # Freeze all layers first
        for param in layer_params:
            param.requires_grad = False

        if train_first_n:
            # Unfreeze the first `n` layers
            count = 0
            for param in layer_params:
                if count < n:
                    param.requires_grad = True
                    count += 1
        else:
            # Unfreeze the layers after the first `n` layers
            count = 0
            for param in layer_params:
                if count >= n:
                    param.requires_grad = True
                    count += 1


    



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes in CIFAR-10

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def load_remaining_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[n:]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)
        temp = self.state_dict()
    
class MLR(nn.Module):
    def __init__(self, args):
        input_size = 784
        output_size = 10
        super(MLR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class SVM(nn.Module):
    def __init__(self):
        input_size = 60
        output_size = 10
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2,2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x

class Model2(nn.Module):
    def __init__(self, args):
        super(Model2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d((2, 2)),
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class CNN1(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(3136, 64)
        self.lin2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x



class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        return self.out(x)

    

class FuseBaseSelfAttention(nn.Module):
    def __init__(self, d_hid: int = 64, d_head: int = 4, is_adapter: bool = False):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head
        
        self.is_adapter = is_adapter
        self.adapter1 = None
        self.adapter2 = None

    def forward(self, x: Tensor, val_a=None, val_b=None, a_len=None):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att)
        att = att.transpose(1, 2)

        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len + val_b[idx]:] = -1e5

        att = torch.softmax(att, dim=2)

        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head * self.d_hid)

        if self.is_adapter:
            x = self.adapter1(x)
            x = self.adapter2(x)

        return x

    def add_adapters(self, adapter_hidden_dim, dropout):
        """添加适配器层"""
        self.adapter1 = AdapterBlock(self.d_hid * self.d_head, adapter_hidden_dim, dropout)
        self.adapter2 = AdapterBlock(self.d_hid * self.d_head, adapter_hidden_dim, dropout)


class AdapterBlock(nn.Module):
    def __init__(self, input_dim, adapter_hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, adapter_hidden_dim)
        self.linear2 = nn.Linear(adapter_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        for module in [self.linear1, self.linear2]:
            nn.init.normal_(module.weight, 0, .01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # down-project
        u = torch.relu(self.linear1(self.dropout(x)))  # (seq_len, B, h)
        # up-project
        u = self.linear2(u)  # (seq_len, B, d)
        # skip connection
        u = x + u
        return u



class MMDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA, 
        modalityB, 
        default_feat_shape_a,
        default_feat_shape_b,
        data_len: int, 
        simulate_feat=None
    ):
        
        self.data_len = data_len
        
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.simulate_feat = simulate_feat
        
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b

        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        
        # modality A, if missing replace with 0s, and mask
        if data_a is not None: 
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else: 
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        # modality B, if missing replace with 0s
        if data_b is not None:
            if len(data_b.shape) == 3: data_b = data_b[0]
            data_b = torch.tensor(data_b)
            len_b = len(data_b)
        else: 
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0
        return data_a, data_b, len_a, len_b, label


class ImageTextClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        img_input_dim: int,     # Image data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6,           # Head dim
        is_adapter: bool=True
    ):
        super(ImageTextClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.is_adapter=is_adapter
        
        # Projection head
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid)
        )
            
        # RNN module
        self.text_rnn = nn.GRU(
            input_size=text_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "fuse_base":
            if is_adapter:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head,
                    is_adapter=self.is_adapter
                )
                self.fuse_att.add_adapters(adapter_hidden_dim = 16, dropout = self.dropout_p)
            else:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head
                )

        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            
        self.init_weight()
        
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_img, x_text, len_i, len_t):
        x_img = self.img_proj(x_img[:, 0, :])
        
        if len_t[0] != 0:
            x_text = pack_padded_sequence(
                x_text, 
                len_t.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )
        x_text, _ = self.text_rnn(x_text)
        if len_t[0] != 0:
            x_text, _ = pad_packed_sequence(x_text, batch_first=True)
        
        if self.en_att:
            if self.att_name == "fuse_base":
                x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
                x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
        else:
            x_text = torch.mean(x_text, axis=1)
            x_mm = torch.cat((x_img, x_text), dim=1)

        preds = self.classifier(x_mm)
        return preds, x_mm
    
    def load_first_n_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[:n]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)



    def load_remaining_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[n:]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)
        temp = self.state_dict()

    def load_adapter_layers(self, w):
        own_state = self.state_dict()
        adapter_layer_names = [name for name in own_state.keys() if 'adapter' in name]
        
        for name, param in w.items():
            if name in adapter_layer_names:
                own_state[name].copy_(param)
        
        self.load_state_dict(own_state, strict=False)    

class BaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1, 1)

    def forward(
        self,
        x: Tensor,
        val_l=None
    ):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att).squeeze(-1)
        if val_l is not None:
            for idx in range(len(val_l)):
                att[idx, val_l[idx]:] = -1e6
        att = torch.softmax(att, dim=1)
        x = (att.unsqueeze(2) * x).sum(axis=1)
        return x
    

class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x
    
    
class HARClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        acc_input_dim: int,     # Acc data input dim
        gyro_input_dim: int,    # Gyro data input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6,           # Head dim
        is_adapter: bool=False
    ):
        super(HARClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.is_adapter=is_adapter
        # Conv Encoder module
        self.acc_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        self.gyro_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.acc_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        self.gyro_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "multihead":
            self.acc_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            
            self.gyro_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.acc_att = BaseSelfAttention(d_hid=d_hid)
            self.gyro_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            if is_adapter:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head,
                    is_adapter=self.is_adapter
                )
                # adapter_hidden_dim
                self.fuse_att.add_adapters(adapter_hidden_dim = 16, dropout = self.dropout_p)
            else:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head
                )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Projection head
            self.acc_proj = nn.Linear(d_hid, d_hid//2)
            self.gyro_proj = nn.Linear(d_hid, d_hid//2)
            
            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        self.init_weight()


    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_acc, x_gyro, l_a, l_b):
        # 1. Conv forward
        x_acc = self.acc_conv(x_acc)
        x_gyro = self.gyro_conv(x_gyro)
        # 2. Rnn forward
        x_acc, _ = self.acc_rnn(x_acc)
        x_gyro, _ = self.gyro_rnn(x_gyro)

        # Length of the signal
        l_a = l_a // 8
        l_b = l_b // 8
        
        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_acc, _ = self.acc_att(x_acc, x_acc, x_acc)
                x_gyro, _ = self.gyro_att(x_gyro, x_gyro, x_gyro)
                # 4. Average pooling
                x_acc = torch.mean(x_acc, axis=1)
                x_gyro = torch.mean(x_gyro, axis=1)
            elif self.att_name == 'base':
                # get attention output
                x_acc = self.acc_att(x_acc)
                x_gyro = self.gyro_att(x_gyro)
            elif self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_acc, x_gyro), dim=1)
                x_mm = self.fuse_att(
                    x_mm, 
                    val_a=l_a, 
                    val_b=l_b, 
                    a_len=x_acc.shape[1]
                )
        else:
            # 4. Average pooling
            x_acc = torch.mean(x_acc, axis=1)
            x_gyro = torch.mean(x_gyro, axis=1)
            x_mm = torch.cat((x_acc, x_gyro), dim=1)

        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            x_acc = self.acc_proj(x_acc)
            x_gyro = self.gyro_proj(x_gyro)
            x_mm = torch.cat((x_acc, x_gyro), dim=1)
        
        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds, x_mm
    
    
    def load_first_n_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[:n]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)
        temp = self.state_dict()



    def load_remaining_layers(self, w, n):
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[n:]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)
        temp = self.state_dict()
        
    def load_adapter_layers(self, w):
        own_state = self.state_dict()

        adapter_layer_names = [name for name in own_state.keys() if 'adapter' in name]
        
        for name, param in w.items():
            if name in adapter_layer_names:
                own_state[name].copy_(param)
        
        self.load_state_dict(own_state, strict=False)    
       

class MMActionClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        audio_input_dim: int,   # Audio feature input dim
        video_input_dim: int,   # Frame-wise video feature input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6,           # Head dim
        is_adapter: bool=True
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.is_adapter = is_adapter
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid,
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        self.video_rnn = nn.GRU(
            input_size=video_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )
        
        # Attention modules
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            
            self.video_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )

        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.video_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            if is_adapter:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head,
                    is_adapter=self.is_adapter
                )
                # adapter_hidden_dim
                self.fuse_att.add_adapters(adapter_hidden_dim = 16, dropout = self.dropout_p)
            else:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head
                )


        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Projection head
            self.audio_proj = nn.Linear(d_hid, d_hid//2)
            self.video_proj = nn.Linear(d_hid, d_hid//2)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            
         # Projection head
        self.init_weight()
        
            
            
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(
        self, 
        x_audio, 
        x_video, 
        len_a, 
        len_v
    ):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        
        # 2. Rnn forward
        # max pooling, time dim reduce by 8 times
        len_a = len_a//8
        if len_a[0] != 0:
            x_audio = pack_padded_sequence(
                x_audio, 
                len_a.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )
        if len_v[0] != 0:
            x_video = pack_padded_sequence(
                x_video, 
                len_v.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )

        x_audio, _ = self.audio_rnn(x_audio) 
        x_video, _ = self.video_rnn(x_video) 
        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(   
                x_audio, 
                batch_first=True
            )
        if len_v[0] != 0:
            x_video, _ = pad_packed_sequence(
                x_video, 
                batch_first=True
            )

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_video, _ = self.video_att(x_video, x_video, x_video)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'additive':
                # get attention output
                x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
                x_video = self.video_att(x_video, x_video, x_video, len_v)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_video), dim=1)
                
                # x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)
                if self.is_adapter:
                    x_mm = self.fuse_att(x_mm, val_a=len_a, val_b=len_v, a_len=a_max_len)  
                else:
                    # FuseBaseSelfAttention expects x and optional val_a, val_b, a_len
                    x_mm = self.fuse_att(x_mm, val_a=len_a, val_b=len_v, a_len=a_max_len)  
                    
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_video = self.video_att(x_video, len_v)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
            x_mm = torch.cat((x_audio, x_video), dim=1)

        # 5. Projection with no attention
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_video = self.video_proj(x_video)
            x_mm = torch.cat((x_audio, x_video), dim=1)
        # 6. MM embedding and predict
        preds = self.classifier(x_mm)
        return preds, x_mm
    
    
    def load_first_n_layers(self, w, n):
        """ 加载前n层的参数 """
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[:n]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)



    def load_remaining_layers(self, w, n):
        """ 加载从第n层到最后的参数 """
        own_state = self.state_dict()
        layer_names = [name for name in own_state.keys()]
        for name, param in w.items():
            if name in layer_names[n:]:
                own_state[name].copy_(param)
        self.load_state_dict(own_state, strict=False)


    def load_adapter_layers(self, w):
        own_state = self.state_dict()

        adapter_layer_names = [name for name in own_state.keys() if 'adapter' in name]
        
        for name, param in w.items():
            if name in adapter_layer_names:
                own_state[name].copy_(param)
        
        self.load_state_dict(own_state, strict=False)

        
        