from model.utils import box_utils

from model.mobilenetv1_ssd_config import image_size, priors

import cv2
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mobilenetv1 import MobileNetV1

num_classes = 21

class SSD(nn.Module):
    def __init__(self, num_classes, backbone, is_training=True, device=None):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone.to(device)
        self.is_training = is_training

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        self.regression_headers = nn.ModuleList([
            nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            # TODO: change to kernel_size=1, padding=0?
        ]).cuda()

        self.classification_headers = nn.ModuleList([
            nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            # TODO: change to kernel_size=1, padding=0?
        ]).cuda()

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not is_training:
            pass
        # if is_test:
        #     self.config = config
        #     self.priors = config.priors.to(self.device)

    def forward(self, x):
        confidences, locations = [], []

        y0, y1 = self.backbone(x)

        confidence, location = self.compute_header(0, y0)
        confidences.append(confidence)
        locations.append(location)
        confidence, location = self.compute_header(1, y1)
        confidences.append(confidence)
        locations.append(location)

        x = self.extras[0].cuda()(y1)
        confidence, location = self.compute_header(2, x)
        confidences.append(confidence)
        locations.append(location)
        x = self.extras[1].cuda()(x)
        confidence, location = self.compute_header(3, x)
        confidences.append(confidence)
        locations.append(location)
        x = self.extras[2].cuda()(x)
        confidence, location = self.compute_header(4, x)
        confidences.append(confidence)
        locations.append(location)
        x = self.extras[3].cuda()(x)
        confidence, location = self.compute_header(5, x)
        confidences.append(confidence)
        locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if not self.is_training:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, priors.cuda(), center_variance=0.1, size_variance=0.2
            )
            boxes = box_utils.center_form_to_corner_form(boxes)

            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        return locations, labels

class SSDLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio, center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(SSDLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, conf_pred, loca_pred, label_true, loca_true):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = conf_pred.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(conf_pred, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, label_true, self.neg_pos_ratio)

        confidence = conf_pred[mask, :]
        classification_loss = torch.nn.CrossEntropyLoss()(confidence.reshape(-1, num_classes), label_true[mask])
        pos_mask = label_true > 0
        loca_pred = loca_pred[pos_mask, :].reshape(-1, 4)
        loca_true = loca_true[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(loca_pred, loca_true, reduction='sum')
        num_pos = loca_true.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos

class Predictor:
    def __init__(self, model, img_size, iou_threshold=0.5, filter_threshold=0.01,
                 candidate_size=200, sigma=0.5, device=None):
        self.model = model
        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.sigma = sigma

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        h, w, _ = image.shape
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)
        images = image.unsqueeze(0)
        images = images.to(self.device)

        with torch.no_grad():
            scores, boxes = self.model(images)

        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold

        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, None,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= w
        picked_box_probs[:, 1] *= h
        picked_box_probs[:, 2] *= w
        picked_box_probs[:, 3] *= h
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

def main():
    x = torch.randn(1, 3, image_size, image_size).cuda()

    backbone = MobileNetV1()
    ssd = SSD(num_classes=num_classes, backbone=backbone, is_training=True).cuda().eval()

    for _ in range(10):
        pre = time.time()
        ssd(x)
        print(time.time() - pre)


if __name__=='__main__':
    main()