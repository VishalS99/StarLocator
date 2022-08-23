import typing as t
import torch.backends.cudnn as cudnn
import numpy as np
import torch
from tqdm import tqdm
from utils import DEVICE, synthesize_data, normalize_data, denormalize_data
from model import StarModel
from custom_loss.oriented_iou_loss import cal_diou


class StarDataset(torch.utils.data.Dataset):
    """
    Return star image and labels
    Return
        - labels of {0/1} if task is classification
        - labels of [x,y,alpha,w,h] for bounding box regression
    """

    def __init__(self, data_size: int = 250000, classification: bool = True):
        self.data_size = data_size
        self.classification = classification

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idxm) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if self.classification:
            image, label = synthesize_data()
            if np.any(np.isnan(label)):
                label = 0
            else:
                label = 1
            return image[None], label
        image, label = synthesize_data(has_star=True)
        return image[None], label


def train(
        model: StarModel,
        loss_criterion,
        optimizer,
        scheduler,
        dl: StarDataset,
        vl: StarDataset,
        num_epochs: int,
        classification: bool) -> StarModel:

    for epoch in range(num_epochs):

        print(f"## EPOCH: {epoch}")

        losses = []
        ious = []
        model.train()

        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float() 
            if classification:
                label = label.to(DEVICE).float()
            else:
                label = normalize_data(label).to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(image, classification)

            if classification:
                loss = loss_criterion(torch.squeeze(preds), label)
            else:
                loss, iou = loss_criterion(
                    torch.unsqueeze(denormalize_data(preds), 1),
                    torch.unsqueeze(denormalize_data(label), 1),
                    'smallest'
                )
                loss = torch.mean(loss)
                iou = torch.mean(iou)
                ious.append(iou.item())
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        scheduler.step()
        print(
            "#- Training Metrics ",
            "- Mean Loss: ",
            np.mean(losses),
            " - Mean IoU: ",
            np.mean(ious) if classification == False else None)

        # Validation
        val_loss = []
        val_iou = []
        model.eval()

        for image, label in tqdm(vl, total=len(vl)):
            image = image.to(DEVICE).float() 
            if classification:
                label = label.to(DEVICE).float()
            else:
                label = normalize_data(label).to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(image, classification)

            if classification:
                loss = loss_criterion(torch.squeeze(preds), label)
            else:
                loss, iou = loss_criterion(
                    torch.unsqueeze(denormalize_data(preds), 1),
                    torch.unsqueeze(denormalize_data(label), 1),
                    'smallest'
                )
                loss = torch.mean(loss)
                iou = torch.mean(iou)
                val_iou.append(iou.item())
            val_loss.append(loss.detach().cpu().numpy())
        print(
            "#- Validation Metrics ",
            " - Mean Loss: ",
            np.mean(val_loss),
            " - Mean IoU: ",
            np.mean(val_iou) if classification == False else [], "\n")

    return model


def main():

    model = StarModel().to(DEVICE)
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Trainable parameters: ", pytorch_total_params)

    cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.1)

    print("Bounding Box Regression Training: \n")
    for param in model.classification.parameters():
        param.requires_grad = False

    criterion_regression = cal_diou

    train_dataloader_reg = torch.utils.data.DataLoader(
        StarDataset(
            data_size=250000,
            classification=False),
        batch_size=128,
        num_workers=16)
    validation_dataloader_reg = torch.utils.data.DataLoader(StarDataset(
        data_size=50000, classification=False), batch_size=64, num_workers=8)

    star_model = train(
        model,
        criterion_regression,
        optimizer,
        scheduler,
        train_dataloader_reg,
        validation_dataloader_reg,
        num_epochs=25,
        classification=False
    )

    print("Star Classification Training: \n")

    for param in star_model.classification.parameters():
        param.requires_grad = True

    for param in star_model.regression.parameters():
        param.requires_grad = False

    for param in star_model.conv1.parameters():
        param.requires_grad = False

    for param in star_model.conv2.parameters():
        param.requires_grad = False

    for param in star_model.conv3.parameters():
        param.requires_grad = False

    for param in star_model.conv4.parameters():
        param.requires_grad = False

    for param in star_model.conv5.parameters():
        param.requires_grad = False

    for param in star_model.conv6.parameters():
        param.requires_grad = False

    for param in star_model.conv7.parameters():
        param.requires_grad = False

    criterion_classification = torch.nn.BCELoss()
    train_dataloader_class = torch.utils.data.DataLoader(StarDataset(
        data_size=250000, classification=True), batch_size=128, num_workers=8)

    validation_dataloader_class = torch.utils.data.DataLoader(StarDataset(
        data_size=50000, classification=True), batch_size=64, num_workers=8)

    star_model = train(
        star_model,
        criterion_classification,
        optimizer,
        scheduler,
        train_dataloader_class,
        validation_dataloader_class,
        num_epochs=2,
        classification=True)

    model_num = 1
    print("Saved model: ", str(model_num))
    torch.save(star_model.state_dict(), "model" + str(model_num) + ".pth")


if __name__ == "__main__":
    main()
