import torch
import random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm
from argparse import ArgumentParser
import os, sys, time, torch, random, argparse, json, copy
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = {
    "tiny-imagenet-200" : 200,
    "OfficeHome" : 65
}

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, num_classes=200, resnet18=False, pretrained=True):
        super(ResNet, self).__init__()
        if resnet18:
            if (pretrained) :
                self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            else :
                self.network = torchvision.models.resnet18()
            self.n_outputs = 512
        else:
            if (pretrained) :
                self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            else :
                self.network = torchvision.models.resnet18()
            self.n_outputs = 2048

        # save memory
        del self.network.fc
        self.network.fc = nn.Identity()

        self.classifier = nn.Linear(self.n_outputs, num_classes)
        self.model = nn.Sequential(self.network, self.classifier)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.model(x)

class Aux_ResNet18(torch.nn.Module):
    def __init__(self, base_model_path, num_classes=200):
        super(Aux_ResNet18, self).__init__()

        if (base_model_path) :
            self.network = torchvision.models.resnet18()
        else :
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        self.network.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

        if (base_model_path) :
            saved_model_state_dict = torch.load(base_model_path)
            network_state_dict = self.network.state_dict()
            classifier_state_dict = self.classifier.state_dict()

            for name, param in saved_model_state_dict.items() :
                if (name.startswith('network')) :
                    actual_name = '.'.join(name.split('.')[1:])
                    network_state_dict[actual_name] = saved_model_state_dict[name]
                if (name.startswith('classifier')) :
                    actual_name = '.'.join(name.split('.')[1:])
                    classifier_state_dict[actual_name] = saved_model_state_dict[name]

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

        return aux_output_1, aux_output_2, aux_output_3, output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class forecasting_classifier(nn.Module):
    def __init__(self, input_size):
        super(forecasting_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class forecasting_classifier_new(nn.Module):

    def __init__(self, input_size, dataset):
        self.dataset = dataset
        self.input_size = input_size
        super(forecasting_classifier_new, self).__init__()
        self.red_fc1 = nn.Linear(input_size, 128)
        self.red_fc2 = nn.Linear(128, 32)
        self.red_fc3 = nn.Linear(32, 8)

        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
          
        x_aux =   x[:, self.input_size*2:self.input_size*3]
        x_main = x[:, self.input_size*3:]

        x_main = self.softmax(x_main)
        #print("Shape of x_main, x_aux:", x_main.shape , x_aux.shape)
        entropy = torch.sum(-x_main * torch.log(x_main), dim=1, keepdim=True)
        confidence_margin = torch.max(x_main, dim=1, keepdim=True)[0] - torch.sort(x_main, dim=1, descending=True)[0][:, 1:2]

        x = x_aux

        x = self.relu(self.red_fc1(x))
        x = self.relu(self.red_fc2(x))
        x = self.relu(self.red_fc3(x))
        x = torch.cat((x, entropy, confidence_margin), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
def set_bn_track_running_stats_false(model):
    for module in model.modules():
        if (isinstance(module, nn.BatchNorm2d)):
            module.track_running_stats = False

def vanilla_distillation_loss (student_output, teacher_output, temperature=4) :
    student_softmax = nn.functional.log_softmax(student_output / temperature, dim=1)
    teacher_softmax = nn.functional.softmax(teacher_output / temperature, dim=1)
    distillation_loss = nn.functional.kl_div(student_softmax, teacher_softmax.detach(), reduction='batchmean')
    return distillation_loss

def evaluate_model(model, testloader, device="cuda:0", aux=False):
    model.eval()
    correct = 0

    correct_1 = 0
    correct_2 = 0
    correct_3 = 0

    total = 0
    model.to(device)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            if not (aux) :
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else :
                output_1, output_2, output_3, output = model(images)
                _, predicted_1 = torch.max(output_1.data, 1)
                _, predicted_2 = torch.max(output_2.data, 1)
                _, predicted_3 = torch.max(output_3.data, 1)
                _, predicted = torch.max(output.data, 1)

                # print(predicted_1, predicted_2, predicted_3, predicted)
                correct_1 += (predicted_1 == labels).sum().item()
                correct_2 += (predicted_2 == labels).sum().item()
                correct_3 += (predicted_3 == labels).sum().item()
                correct += (predicted == labels).sum().item()

                total += labels.size(0)

    if (aux) :
        return correct_1/total, correct_2/total, correct_3/total, correct/total
    else :
        return correct/total

def evaluate_forecasting_model(dataset, model, forecasting_model, testloader):
    model.eval()
    forecasting_model.eval()

    correct = 0
    total = 0
    print(dataset)
    random_correct = 0
    constant_correct = 0

    

    with torch.no_grad():
        for images, labels in testloader:
            cuda0 = torch.device('cuda:0')
            random_labels = torch.randint(0, num_classes[dataset], labels.shape,device=cuda0)
            labels = labels.to("cuda:0")
            print(random_labels.get_device() , labels.get_device())
            
            random_correct += (random_labels == labels).sum()
            random_correct = random_correct.cpu().item()

            constant_labels = (torch.zeros(labels.shape))
            constant_labels = constant_labels.to("cuda:0")
            constant_correct += (constant_labels == labels).sum()
            constant_correct= constant_correct.cpu().item()

            output_1, output_2, output_3, output = model.predict(images)

            forecasting_input = torch.cat((output_1, output_2, output_3, output), dim=1)
            forecasting_output = forecasting_model(forecasting_input)

            correctness = (torch.argmax(output, dim=1) == labels)
            correctness = correctness.type(torch.int64)

            predicted_correctness = torch.argmax(forecasting_output, dim=1)

            total += labels.size(0)
            correct += (predicted_correctness == correctness).sum().item()

    print("Random accuracy :", random_correct/total)
    print("Constant label accuracy :", constant_correct/total)
    return correct/total

def train_model(model, trainloader, testloader, optimizer, num_epochs=10, save_path=None, device="cuda:0", aux=False) :
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if (aux) :
        model.network.eval()
        model.classifier.eval()
        set_bn_track_running_stats_false(model.network)

    for epoch in range(num_epochs) :
        for images, labels in tqdm(trainloader) :
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if (aux) :
                output_1, output_2, output_3, output = model(images)
                loss =  criterion(output_1, labels) + criterion(output_2, labels) + criterion(output_3, labels)  
                # print(loss)
            else :
                model_output = model(images)
                loss = criterion(model_output, labels)
            loss.backward()
            optimizer.step()

        accuracy = evaluate_model(model, testloader, device=device, aux=aux)
        print(f"Epoch : {epoch + 1} Model accuracy : {accuracy}")


        if (save_path) :
            name = save_path.split('.')[0]
            model_save_path = name + f"_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_save_path)

    model.eval()

def save_forecasting_dataset(model, forecasting_trainloader, device="cuda:0") :
    model.to(device)
    model.eval()
    set_bn_track_running_stats_false(model.network)

    inputs = []
    outputs = []

    with torch.no_grad() :
        for images, labels in tqdm(forecasting_trainloader) :
            images, labels = images.to(device), labels.to(device)
            output_1, output_2, output_3, output = model(images)

            forecasting_input = torch.cat((output_1, output_2, output_3, output), dim=1)
            correctness = (torch.argmax(output, dim=1) == labels)
            correctness = correctness.type(torch.int64)

            inputs.extend(list(forecasting_input))
            outputs.extend(list(correctness))

    dataset = list(zip(inputs, outputs))

    # Save dataset to a file using torch.save
    torch.save(dataset, '/content/drive/MyDrive/models/dataset_new.pth')

def train_forecasting_model( minibatches, algorithm , forecasting_model, uda_device,  forecasting_optimizer, dataset ) :
    all_x = torch.cat([x for x, y in minibatches])
    all_y = torch.cat([y for x, y in minibatches])

    
    
    criterion = FocalLoss()

    forecasting_optimizer.zero_grad()
    output_1, output_2, output_3, output = algorithm.predict(all_x)

    forecasting_input = torch.cat((output_1, output_2, output_3, output), dim=1)
    forecasting_output = forecasting_model(forecasting_input)

    correctness = (torch.argmax(output, dim=1) == all_y)
    correctness = correctness.type(torch.int64)

    predicted_correctness = torch.argmax(forecasting_output, dim=1)
    loss = criterion(forecasting_output, correctness)
    print(loss, torch.sum(correctness)/correctness.shape[0], predicted_correctness)

    loss.backward()
    forecasting_optimizer.step()
    
    accuracy = evaluate_forecasting_model(dataset, algorithm, forecasting_model, uda_device)
    print(f" Forecasting Model accuracy : {accuracy}")

    forecasting_model.eval()
    """
    if (save_path) :
        torch.save(forecasting_model.state_dict(), save_path)
    """

def vanilla_distillation (teacher_model, student_model, trainloader, testloader, optimizer, num_epochs=10, save_path=None, device="cuda:0") :
    criterion = nn.CrossEntropyLoss()

    teacher_model.to(device)
    student_model.to(device)

    ## Fixing the teacher model
    teacher_model.eval()
    set_bn_track_running_stats_false(teacher_model)

    for epoch in range(num_epochs):
        student_model.train()

        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            student_output = student_model(images)
            teacher_output = teacher_model(images)

            ce_loss = criterion(student_output, labels)
            distillation_loss = vanilla_distillation_loss(student_output, teacher_output)

            total_loss = ce_loss + 0.5 * distillation_loss
            total_loss.backward()
            optimizer.step()

        teacher_accuracy = evaluate_model(teacher_model, testloader, device=device)
        student_accuracy = evaluate_model(student_model, testloader, device=device)

        print(f"Epoch : {epoch + 1} Train accuracy : {teacher_accuracy} Test accuracy : {student_accuracy}")

        if (save_path) :
            name = save_path.split('.')[0]
            model_save_path = name + f"_{epoch + 1}.pth"
            torch.save(student_model.state_dict(), model_save_path)

def Nurips (dataset,teacher_model, student_model, trainloader, testloader, forecasting_trainloader, optimizer, num_epochs=100, save_path=None, device="cuda:0", L= 10) :
    criterion = nn.CrossEntropyLoss()

    input_size = 4*num_classes[dataset]
    forecasting_model = forecasting_classifier_new(input_size,dataset)


    teacher_model.to(device)
    student_model.to(device) 
    forecasting_model.to(device)

    ## Fixing the teacher model
    teacher_model.eval()
    set_bn_track_running_stats_false(teacher_model)
    weights =[]

            
    for epoch in range(num_epochs):
        student_model.train()

        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            student_output, aux1, aux2, aux3 = student_model(images)
            teacher_output = teacher_model(images)

           
            if epoch%L==0:
                loss = criterion(aux1, labels)+ criterion(aux2, labels)+ criterion(aux3, labels)
                loss.backward()
                optimizer.step()

                train_forecasting_model(dataset, student_model, forecasting_model, forecasting_trainloader, optimizer, num_epochs=10, save_path=None, device="cuda:0")
                 

            forecasting_input = torch.cat((aux1, aux2, aux3), dim=1)
            weights = forecasting_model(forecasting_input)

            ce_loss = criterion(student_output, labels)
            distillation_loss = vanilla_distillation_loss(student_output, teacher_output)

            total_loss = weights*ce_loss + 0.5 * distillation_loss
            total_loss.backward()
            optimizer.step()

        teacher_accuracy = evaluate_model(teacher_model, testloader, device=device)
        student_accuracy = evaluate_model(student_model, testloader, device=device)

        print(f"Epoch : {epoch + 1} Train accuracy : {teacher_accuracy} Test accuracy : {student_accuracy}")

        if (save_path) :
            name = save_path.split('.')[0]
            model_save_path = name + f"_{epoch + 1}.pth"
            torch.save(student_model.state_dict(), model_save_path)


def main(args): 

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    ])


    
    train_path = args.data_path+ "/train"
    test_path = args.data_path+ "/test"
    
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)

    
    train_size = len(trainset)
    train_set, forecasting_train_set = torch.utils.data.random_split(trainset, [train_size*0.9, train_size*0.1])
    train_set, valset = torch.utils.data.random_split(trainset, [train_size*0.9*0.9, train_size*0.9*0.1])
    forecasting_train_set, forecasting_valset = torch.utils.data.random_split(forecasting_train_set [train_size*0.1*0.9, train_size*0.1*0.1])
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    forecasting_trainloader = torch.utils.data.DataLoader(forecasting_train_set, batch_size=32, shuffle=True, num_workers=2)
    forecasting_valloader = torch.utils.data.DataLoader(forecasting_valset, batch_size=32, shuffle=True, num_workers=2)
    


    if (args.train_type == 0) :
        teacher_model = ResNet(num_classes = num_classes[args.dataset])
        teacher_optimizer = torch.optim.SGD(teacher_model.parameters(), lr=args.lr, momentum=args.momentum)

        train_model(teacher_model, trainloader, val_loader, teacher_optimizer, \
                        num_epochs=args.num_epochs, \
                        save_path=args.save_path, \
                        device=args.device)
    elif (args.train_type == 1) :
        teacher_model = ResNet(num_classes = num_classes[args.dataset])
        teacher_model.load_state_dict(torch.load(args.teacher_model_path))

        student_model = ResNet(num_classes = num_classes[args.dataset],resnet18=True)
        
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum)
        vanilla_distillation(teacher_model, \
                            student_model, \
                            trainloader, val_loader, \
                            student_optimizer, \
                            num_epochs=args.num_epochs, \
                            save_path=args.save_path, \
                            device=args.device)
    elif (args.train_type == 2) :
        student_model = Aux_ResNet18(base_model_path=args.student_model_path,num_classes = num_classes[args.dataset])
        aux_optimizer = torch.optim.SGD([
                {'params': student_model.aux_conv_1.parameters()},
                {'params': student_model.aux_fc_1.parameters()},
                {'params': student_model.aux_conv_2.parameters()},
                {'params': student_model.aux_fc_2.parameters()},
                {'params': student_model.aux_conv_3.parameters()},
                {'params': student_model.aux_fc_3.parameters()}
            ],
            lr=0.01,
            momentum=0.9
        )
        train_model(student_model, trainloader, val_loader, \
                    aux_optimizer, num_epochs=args.num_epochs, \
                    save_path=args.save_path, \
                    device=args.device, aux=True)
        
    elif (args.train_type == 3) :
        student_model = Aux_ResNet18(base_model_path="",num_classes = num_classes[args.dataset])
        student_model.load_state_dict(torch.load(args.student_model_path))

        forecasting_model = forecasting_classifier(600)
        forecasting_optimizer = torch.optim.SGD(forecasting_model.parameters(), lr=args.lr, momentum=args.momentum)
        train_forecasting_model(args.dataset, student_model, forecasting_model, forecasting_trainloader ,forecasting_valloader,  forecasting_optimizer, \
                                num_epochs=args.num_epochs, save_path=args.save_path, \
                                device=args.device)

        save_forecasting_dataset(student_model, forecasting_trainloader, args.device)


    elif (args.train_type == 4):
        teacher_model = ResNet(num_classes = num_classes[args.dataset])
        student_model = ResNet(num_classes = num_classes[args.dataset],resnet18=True)
        teacher_model.load_state_dict(torch.load(args.teacher_model_path))
        student_model.load_state_dict(torch.load(args.student_model_path))
        optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)
        Nurips (args.dataset, teacher_model, student_model, trainloader, val_loader, forecasting_trainloader,forecasting_valloader, \
                optimizer, num_epochs=args.num_epochs, \
                save_path=args.save_path, device=args.device)
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    
    parser.add_argument("--train_type", type=int, default=0, help="method to be used")
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='', help="The dataset name.")
    parser.add_argument("--teacher_model_path", type=str, default='', help="The dataset name.")
    parser.add_argument("--student_model_path", type=str, default='', help="The dataset name.")
    parser.add_argument("--save_path", type=str, default='', help="The dataset name.")
    parser.add_argument("--momentum", type=int, default=0.9, help="The dataset name.")
    parser.add_argument("--lr", type=int, default=0.01, help="The dataset name.")
    parser.add_argument("--num_epochs", type=int, default=200, help="The dataset name.")
    


    args = parser.parse_args()

    main(args)

