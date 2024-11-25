import torch
import torchvision.transforms as transforms
from torchvision.datasets.imagenet import ImageNet
import os
import numpy as np
import gc
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.bounding_box import getBoudingBox_multi, box_to_seg
import torch.utils.model_zoo as model_zoo

# datasets
from Datasets.ILSVRC import ImageNetDataset_val

# models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import timm

# methods
from Methods.AGCAM.AGCAM import AGCAM, BetterAGCAM

import csv
from csv import DictWriter


class csv_utils:
    def __init__(self, fileName):
        self.fileName = fileName
        self.fieldNames = ["label", "pixel_acc", "iou", "dice", "precision", "recall"]

    def writeFieldName(self):
        with open(self.fileName, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldNames)
            writer.writeheader()
            csvfile.close()

    def appendResult(self, img_name, pixel_acc, iou, dice, precision, recall):
        with open(self.fileName, "a") as csvfile:
            writer = DictWriter(csvfile, fieldnames=self.fieldNames)
            writer.writerow(
                {
                    "label": img_name,
                    "pixel_acc": pixel_acc.item(),
                    "iou": iou.item(),
                    "dice": dice.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                }
            )
            csvfile.close()


class Evaluator:

    def __init__(
        self,
        seed=777,
        img_size=224,
        threshold=0.5,
        root_dir="./ILSVRC/",
    ):
        self.img_size = img_size
        self.threshold = threshold
        self.seed = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        self.validset = ImageNetDataset_val(
            root_dir=self.root_dir,
            transforms=self.transform,
        )

        self.validloader = DataLoader(
            dataset=self.validset,
            batch_size=1,
            shuffle=False,
        )

    def get_method(self, method_name):
        MODEL = "vit_base_patch16_224"
        state_dict = model_zoo.load_url(
            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
            progress=True,
            map_location=self.device,
        )
        class_num = 1000

        model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to(
            self.device
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        if method_name == "AGCAM":
            method = AGCAM(model)
        if method_name == "BetterAGCAM":
            method = BetterAGCAM(model)

        return method

    def doEvaluate(self, method_name):
        method = self.get_method(method_name)
        with torch.enable_grad():
            num_img = 0

            export_file = method_name + "_results.csv"
            csvUtils = csv_utils(export_file)
            csvUtils.writeFieldName()
            for data in tqdm(self.validloader):
                image = data["image"].to("cuda")
                label = data["label"].to("cuda")
                bnd_box = data["bnd_box"].to("cuda").squeeze(0)

                prediction, mask = method.generate(image)

                # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
                if prediction != label:
                    continue
                mask = mask.reshape(1, 1, 14, 14)

                # Reshape the mask to have the same size with the original input image (224 x 224)
                upsample = torch.nn.Upsample(224, mode="bilinear", align_corners=False)
                mask = upsample(mask)

                # Normalize the heatmap from 0 to 1
                mask = (mask - mask.min()) / (mask.max() - mask.min())

                # To avoid the overlapping problem of the bounding box labels, we generate a 0-1 segmentation mask from the bounding box label.
                seg_label = box_to_seg(bnd_box).to(self.device)

                # From the generated heatmap, we generate a bounding box and then convert it to a segmentation mask to compare with the bounding box label.
                mask_bnd_box = getBoudingBox_multi(mask, threshold=self.threshold).to(
                    self.device
                )
                seg_mask = box_to_seg(mask_bnd_box).to(self.device)

                output = seg_mask.view(
                    -1,
                )
                target = seg_label.view(
                    -1,
                ).float()

                tp = torch.sum(output * target)  # True Positive
                fp = torch.sum(output * (1 - target))  # False Positive
                fn = torch.sum((1 - output) * target)  # False Negative
                tn = torch.sum((1 - output) * (1 - target))  # True Negative
                eps = 1e-5
                pixel_acc_ = (tp + tn + eps) / (tp + tn + fp + fn + eps)
                dice_ = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                precision_ = (tp + eps) / (tp + fp + eps)
                recall_ = (tp + eps) / (tp + fn + eps)
                iou_ = (tp + eps) / (tp + fp + fn + eps)
                num_img += 1

                csvUtils.appendResult(
                    data["filename"][0], pixel_acc_, iou_, dice_, precision_, recall_
                )
