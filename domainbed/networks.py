# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from lib import wide_resnet
import copy


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class AuxResNet18(nn.Module):
    def __init__(self, num_classes):
        super(AuxResNet18, self).__init__()
        
        self.base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Assume the pre-trained model is saved in a file named 'resnet18_pretrained.pkl'

        print(torch.load('/home1/durga/sainath/Robust Distillation/KD/train_output/TerraIncognita/TerraIncognita_ERM_Student.pkl')["model_dict"].keys())
        self.base_model.load_state_dict(torch.load('/home1/durga/sainath/Robust Distillation/KD/train_output/TerraIncognita/TerraIncognita_ERM_Student.pkl')["model_dict"])

        # Remove the fully connected layer (classification layer)
        self.base_model.fc = nn.Identity()

        # Define auxiliary layers to be added
        self.aux1 = nn.Linear(64, num_classes)  #output of the first block is 64 features
        self.aux2 = nn.Linear(128, num_classes)  #output of the second block is 128 features
        self.aux3 = nn.Linear(256, num_classes)  #output of the third block is 256 features

        # Main classifier layer that will be added after the final block
        self.classifier = nn.Linear(512, num_classes)  # Assuming output of the final block is 512 features

    def forward(self, x):
        # Forward pass through the base model up to each auxiliary layer
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        # First block
        x1 = self.base_model.layer1(x)
        aux1_out = self.aux1(torch.flatten(x1, 1))

        # Second block
        x2 = self.base_model.layer2(x1)
        aux2_out = self.aux2(torch.flatten(x2, 1))

        # Third block
        x3 = self.base_model.layer3(x2)
        aux3_out = self.aux3(torch.flatten(x3, 1))

        # Final block and main classifier
        x4 = self.base_model.layer4(x3)
        main_out = self.classifier(torch.flatten(x4, 1))

        return main_out, aux1_out, aux2_out, aux3_out

class Aux_ResNet18(torch.nn.Module):
    def __init__(self, num_classes, base_model_path):
        super(Aux_ResNet18, self).__init__()
        
        if (base_model_path) : 
            self.network = torchvision.models.resnet18()
        else : 
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        # Remove the fully connected layer (classification layer)
        self.network.fc = nn.Identity()

        # Main classifier after the final block
        self.classifier = nn.Linear(512, num_classes)

        if (base_model_path) : 
            saved_model_state_dict = torch.load(base_model_path)["model_dict"]
            network_state_dict = self.network.state_dict()
            classifier_state_dict = self.classifier.state_dict()

            for name, param in saved_model_state_dict.items() : 
                if (name.startswith('featurizer')) :
                    tokens = name.split('.')
                    actual_param_name = '.'.join(tokens[2:])
                    network_state_dict[actual_param_name] = saved_model_state_dict[name]
                if (name.startswith('classifier')) :
                    actual_param_name = name.split('.')[-1]
                    classifier_state_dict[actual_param_name] = saved_model_state_dict[name]
                    
            ## Loading saved weights
            self.network.load_state_dict(network_state_dict)
            self.classifier.load_state_dict(classifier_state_dict)

        # Define auxiliary layers to be added
        self.aux_conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=4, padding=1)
        self.aux_avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.aux_fc_1 = nn.Linear(64, num_classes)  
        
        self.aux_conv_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=4, padding=1)
        self.aux_avg_pool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.aux_fc_2 = nn.Linear(128, num_classes)  

        self.aux_conv_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.aux_avg_pool_3 = nn.AdaptiveAvgPool2d((1, 1))
        self.aux_fc_3 = nn.Linear(256, num_classes)  


    def forward(self, x):
        # Forward pass through the base model up to each auxiliary layer
        x = self.network.maxpool(self.network.relu(self.network.bn1(self.network.conv1(x))))

        # First block
        output_1 = self.network.layer1(x)
        aux_output_1 = self.aux_avg_pool_1(self.aux_conv_1(output_1))
        aux_output_1 = self.aux_fc_1(torch.flatten(aux_output_1, 1))

        # Second block
        output_2 = self.network.layer2(output_1)
        aux_output_2 = self.aux_avg_pool_2(self.aux_conv_2(output_2))
        aux_output_2 = self.aux_fc_2(torch.flatten(aux_output_2, 1))

        # Third block
        output_3 = self.network.layer3(output_2)
        aux_output_3 = self.aux_avg_pool_3(self.aux_conv_3(output_3))
        aux_output_3 = self.aux_fc_3(torch.flatten(aux_output_3, 1))

        # Final block and main classifier
        output = self.network.layer4(output_3)
        output = self.network.avgpool(output)
        output = self.classifier(torch.flatten(output, 1))
        
        aux_output_1.requires_grad_()
        aux_output_2.requires_grad_()
        aux_output_3.requires_grad_()
        output.requires_grad_()

        return aux_output_1, aux_output_2, aux_output_3, output

class forecasting_classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(forecasting_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            # self.network = torchvision.models.resnet18()
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
