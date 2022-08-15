from __future__ import print_function, division

import copy
import os
import time

import mlflow
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../dataset/pizza_not_pizza_split'
lr = 0.001
momentum = 0.9
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def get_data_loaders_and_image_datasets():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders, image_datasets


def get_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    return model_ft


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    mlflow.log_param("epochs", num_epochs)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            mlflow.log_metric(f"{phase}_accuracy", float(epoch_acc), step=epoch + 1)
            mlflow.log_metric(f"{phase}_loss", float(epoch_loss), step=epoch + 1)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_model(model):
    filename = "resnet18_pizza_cls_model.pth"
    print(f"Saving model to {filename}")
    torch.save(model.state_dict(), filename)
    mlflow.log_artifact(filename)
    print("Model saved")


def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = 'pizza_classification'
    mlflow.set_experiment(experiment_name)
    mlflow.log_param("dataset", data_dir)


def get_loss_optimizer_scheduler(model):
    # Observe that all parameters are being optimized

    mlflow.log_param("lr", lr)
    mlflow.log_param("momentum", momentum)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler_step_size = 7
    lr_scheduler_gamma = 0.1
    mlflow.log_param("lr_scheduler_step_size", lr_scheduler_step_size)
    mlflow.log_param("lr_scheduler_gamma", lr_scheduler_gamma)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
    criterion = nn.CrossEntropyLoss()
    return criterion, optimizer, scheduler


def main():
    init_mlflow()
    model = get_model()
    criterion, optimizer, scheduler = get_loss_optimizer_scheduler(model)
    data_loaders, image_datasets = get_data_loaders_and_image_datasets()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    train_model(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, num_epochs=2)
    save_model(model)


if __name__ == '__main__':
    main()
