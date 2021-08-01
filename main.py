from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import Backbone

writer = SummaryWriter('/home/jhj/Desktop/JHJ/git/yolov4_backbone_pre-training/runs/1')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# hyper-parameters #####
trainset_rate = 0.9  # 0 ~ 1
epochs = 120
learning_rate = 0.001
batch_size = 32
pretrained_weights_path = "/home/jhj/Desktop/JHJ/git/yolov4_backbone_pre-training/yolov4.conv.137.pth"
classes = ['blossom_end_rot',
           'graymold',
           'powdery_mildew',
           'spider_mite',
           'spotting_disease',
           'snails_and_slugs'
           ]
########################

trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])

dataset = torchvision.datasets.ImageFolder(root='/home/jhj/Desktop/JHJ/projects/data/paprika_6classes/for_imagefolder',
                                           transform=trans)

train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * trainset_rate),
                                                    len(dataset) - int(len(dataset) * trainset_rate)],
                                          generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          )
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        )

net = Backbone(yolov4conv137weight='/home/jhj/Desktop/JHJ/projects/data/yolov4.conv.137.pth')
net.to(device)


def one_hot(labels, num_classes):
    mold = torch.zeros((len(labels), num_classes))
    mold[range(len(labels)), labels] = 1

    return mold


class FocalLossV1(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


criterion = FocalLossV1()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
average_accuracy_list = []

for epoch in range(epochs):
    net.train()

    batch_pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), unit="batch")
    for i, data in batch_pbar:
        inputs, labels = data
        labels = one_hot(labels, len(classes))
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch == 0 and i == 0:
            first_loss = loss.item()
        if i == 0:
            start_loss = loss.item()

        # print a status
        # first_description = '\n%9s%12s%32s' % ("Epoch", "First Loss", "Epoch Start Loss->Present Loss")
        # second_description = f"\n{str(epoch) + '/' + str(epochs - 1):>9}" \
        #                      f" {first_loss:>11.4g}" \
        #                      f"  {start_loss:^16.4g}->{loss.item():^12.4g}"
        # batch_pbar.set_description(first_description + second_description)
        batch_pbar.set_description(
            f"Epoch: {epoch+1}/{epochs}, First Loss: {first_loss:.4g}, "
            f"Start Loss->Running Loss: {start_loss:.4g}->{loss.item():.4g}")

    save_path = f"/home/jhj/Desktop/JHJ/projects/data/yolov4_backbone_pth/backbone_{epoch + 1}.pth"
    torch.save(net.state_dict(), save_path)

    # pretrained = torch.load(f"/home/jhj/Desktop/JHJ/projects/data/yolov4_backbone_pth/backbone_{epoch + 1}.pth")
    # net_dict = net.state_dict()
    # net_dict.update(pretrained)
    # net.load_state_dict(net_dict)

    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    with torch.no_grad():
        for data in val_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            outputs = torch.sigmoid(outputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy_sum = 0
    for i in range(len(classes)):
        temp = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], temp))
        accuracy_sum += temp
    print("correct:", class_correct)
    print(" total :", class_total)
    print('Accuracy average: ', accuracy_sum / len(classes))
    average_accuracy_list.append(accuracy_sum)
print(max(average_accuracy_list), average_accuracy_list.index(max(average_accuracy_list)))