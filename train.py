import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import Backbone
from loss import FocalLossV1

writer = SummaryWriter('/home/jhj/Desktop/JHJ/git/yolov4_backbone_pre-training/runs/1')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# hyper-parameters #####
trainset_rate = 0.9  # 0 ~ 1
epochs = 200
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


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img = img.to("cpu")
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds_tensor = preds_tensor.to("cpu")
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])

dataset = torchvision.datasets.ImageFolder(root='/home/jhj/Desktop/JHJ/projects/data/paprika_6classes/for_imagefolder',
                                           transform=trans)

train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * trainset_rate),
                                                    len(dataset) - int(len(dataset) * trainset_rate)],
                                          generator=torch.Generator().manual_seed(25))
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          )
val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        )

net = Backbone()
net.to(device)


def one_hot(labels, num_classes):
    mold = torch.zeros((len(labels), num_classes))
    mold[range(len(labels)), labels] = 1

    return mold


criterion = FocalLossV1()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
accuracy_set = {0}
average_accuracy_list = []
running_loss = 0.0

for epoch in range(epochs):
    net.train()

    batch_pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), unit="batch")
    for i, data in batch_pbar:
        input, label = data
        labels = one_hot(label, len(classes))
        inputs, labels = input.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch == 0 and i == 0:
            first_loss = loss.item()
        if i == 0:
            start_loss = loss.item()

        batch_pbar.set_description(
            f"Epoch: {epoch + 1}/{epochs}, First Loss: {first_loss:.4g}, "
            f"Start Loss->Running Loss: {start_loss:.4g}->{loss.item():.4g}")

        running_loss += loss.item()
        if i % 50 == 49:
            writer.add_scalar('training loss', running_loss / 50, epoch * len(train_loader) + i)
            writer.add_figure('predictions vs actuals', plot_classes_preds(net, inputs, label),
                              global_step=epoch * len(train_loader) + i)
            running_loss = 0.0

    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in val_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            outputs = torch.sigmoid(outputs)
            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            _, class_preds_batch = torch.max(outputs, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)

    accuracy_sum = 0
    for i in range(len(classes)):
        temp = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], temp))
        accuracy_sum += temp

    if int(accuracy_sum / len(classes)) >= max(accuracy_set):
        accuracy_set.add(int(accuracy_sum / len(classes)))
        save_path = f"/home/jhj/Desktop/JHJ/projects/data/yolov4_backbone_pth/backbone_{epoch + 1}.pth"
        torch.save(net.state_dict(), save_path)

    print("correct:", class_correct)
    print(" total :", class_total)
    print('Accuracy average: ', accuracy_sum / len(classes))
    average_accuracy_list.append(accuracy_sum / len(classes))

print(max(average_accuracy_list), average_accuracy_list.index(max(average_accuracy_list)) + 1)
