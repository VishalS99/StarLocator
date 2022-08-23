import numpy as np
import torch
from tqdm import tqdm
from model import StarModel
from utils import DEVICE, score_iou, synthesize_data, denormalize_data


def load_model():
    model = StarModel()
    model = model.to(DEVICE)
    model.load_state_dict(torch.load("model12_32.pth"))
    model.eval()
    return model


def eval(*, n_examples: int = 1024) -> None:
    model = load_model()
    score = []
    print("Runing Model Evaluation: ")
    for _ in tqdm(range(n_examples)):
        image, label = synthesize_data()
        
        with torch.no_grad():
            image = torch.Tensor(image[None,None]).to(DEVICE).float()
            predict = model.pred(image)

        if torch.any(torch.isnan(predict)) == True:
            np_pred = np.squeeze(predict.detach().cpu().numpy())
        else:
            np_pred = np.squeeze(denormalize_data(predict[None]).detach().cpu().numpy())
            (x,y,w,h,alpha) = np_pred
            np_pred = np.array([x,y,alpha,w,h])
        score.append(score_iou(np_pred, label))

    ious = np.asarray(score, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    print("- IoU > 0.5 :", (ious > 0.5).mean(), "\n- IoU > 0.7: ", (ious > 0.7).mean(), "\n- Mean IoU: ", ious.mean())


if __name__ == "__main__":
    eval()
