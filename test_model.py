import matplotlib.pyplot as plt
from utils import _corners, synthesize_data, DEVICE, denormalize_data
import numpy as np
from compute_score import load_model
import torch


def plot(ax, image, label, text):
    ax.imshow(image, cmap="gray")
    ax.set_title(text)
    if label.size > 0:
        bbox = _corners(*label)
        ax.fill(bbox[:, 0], bbox[:, 1], facecolor="none", edgecolor="r")


def prettify_np(a):
    return list(map(lambda x: int(x) if x.is_integer()
                else round(x, 2), a.tolist()))


def get_bbox(model, image, label):
    image1 = torch.Tensor(image[None, None]).to(DEVICE).float()
    predict = model.pred(image1)
    if torch.any(torch.isnan(predict)):
        np_pred = predict.detach().cpu().numpy()
        np_pred = np.squeeze(np_pred)
    else:
        np_pred = denormalize_data(predict[None]).detach().cpu().numpy()
        np_pred = np.squeeze(np_pred)
        (x, y, w, h, alpha) = np_pred
        np_pred = np.array([x, y, alpha, w, h])
    return np_pred


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 3, figsize=(18, 5))
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
    model = load_model()

    image, label = synthesize_data(has_star=True)
    predicted = get_bbox(model, image, label)
    plot(
        ax[0][0],
        image,
        predicted,
        f"star with predicted label {prettify_np(predicted)}")
    plot(ax[1][0], image, label, f"star with label {prettify_np(label)}")

    image, label = synthesize_data(has_star=False)
    predicted = get_bbox(model, image, label)
    plot(ax[0][1], image, predicted, "no star (predicted)")
    plot(ax[1][1], image, label, "no star")

    image, label = synthesize_data(has_star=True, noise_level=0.1)
    predicted = get_bbox(model, image, label)
    plot(
        ax[0][2],
        image,
        predicted,
        f"star (less noise) with predicted label {prettify_np(predicted)}")
    plot(
        ax[1][2],
        image,
        label,
        f"star (less noise) with label {prettify_np(label)}")

    fig.savefig("final_image.png")
