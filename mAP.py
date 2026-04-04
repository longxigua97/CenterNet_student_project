import json
import os
import torch
import torchvision
from tqdm import tqdm
from collections import defaultdict

import config, utils
from model import Centernet_model_ResNet50
from dataloader import CenternetDataset
from predict import predict_box


def find_eval_split(data_folder):
    """Return an available split name for evaluation in priority order."""
    candidates = ("TEST", "TRAINVAL", "TRAIN")
    for split in candidates:
        image_json = os.path.join(data_folder, f"{split}_images.json")
        object_json = os.path.join(data_folder, f"{split}_objects.json")
        if os.path.isfile(image_json) and os.path.isfile(object_json):
            return split
    raise FileNotFoundError(
        f"No valid eval split found in {data_folder}. "
        f"Expected one of: TEST/TRAINVAL/TRAIN JSON pairs."
    )


def compute_ap(recall, precision):
    mrec = torch.cat((torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)))
    mpre = torch.cat((torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)))

    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    indices = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, n_classes, device, iou_threshold=0.5):
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)

    average_precisions = torch.zeros((n_classes,), dtype=torch.float32, device=device)

    for class_idx in range(n_classes):
        true_class_boxes = []
        true_class_difficulties = []
        gt_image_index_map = defaultdict(list)

        global_gt_idx = 0
        for i in range(len(true_labels)):
            mask = true_labels[i] == class_idx
            if mask.sum().item() == 0:
                continue

            boxes = true_boxes[i][mask]
            diffs = true_difficulties[i][mask]
            true_class_boxes.append(boxes)
            true_class_difficulties.append(diffs)

            num_objs = boxes.size(0)
            gt_image_index_map[i].extend(list(range(global_gt_idx, global_gt_idx + num_objs)))
            global_gt_idx += num_objs

        if len(true_class_boxes) == 0:
            continue

        true_class_boxes = torch.cat(true_class_boxes, dim=0)
        true_class_difficulties = torch.cat(true_class_difficulties, dim=0)

        det_class_boxes = []
        det_class_scores = []
        det_class_images = []

        for i in range(len(det_labels)):
            mask = det_labels[i] == class_idx
            if mask.sum().item() == 0:
                continue
            det_class_boxes.append(det_boxes[i][mask])
            det_class_scores.append(det_scores[i][mask])
            det_class_images.extend([i] * mask.sum().item())

        if len(det_class_boxes) == 0:
            continue

        det_class_boxes = torch.cat(det_class_boxes, dim=0)
        det_class_scores = torch.cat(det_class_scores, dim=0)
        det_class_images = torch.tensor(det_class_images, device=device)

        det_class_scores, sort_ind = det_class_scores.sort(dim=0, descending=True)
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros(det_class_boxes.size(0), dtype=torch.float32, device=device)
        false_positives = torch.zeros(det_class_boxes.size(0), dtype=torch.float32, device=device)
        gt_matched = torch.zeros(true_class_boxes.size(0), dtype=torch.uint8, device=device)

        for d in range(det_class_boxes.size(0)):
            image_idx = det_class_images[d].item()
            pred_box = det_class_boxes[d]

            gt_indices = gt_image_index_map.get(image_idx)
            if gt_indices is None:
                false_positives[d] = 1
                continue

            gt_indices_tensor = torch.tensor(gt_indices, device=device)
            gt_boxes_in_image = true_class_boxes[gt_indices_tensor]

            overlaps = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes_in_image).squeeze(0)
            best_iou, max_idx = overlaps.max(dim=0)
            best_gt_idx = gt_indices[max_idx.item()]

            if best_iou >= iou_threshold:
                if true_class_difficulties[best_gt_idx] == 0:
                    if gt_matched[best_gt_idx] == 0:
                        true_positives[d] = 1
                        gt_matched[best_gt_idx] = 1
                    else:
                        false_positives[d] = 1
            else:
                false_positives[d] = 1

        cum_tp = torch.cumsum(true_positives, dim=0)
        cum_fp = torch.cumsum(false_positives, dim=0)

        n_easy = (true_class_difficulties == 0).sum().float().clamp(min=1.0)
        recall = cum_tp / n_easy
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)

        average_precisions[class_idx] = compute_ap(recall, precision)

    return average_precisions


def load_difficulties(data_folder, split):
    object_path = os.path.join(data_folder, f'{split}_objects.json')
    with open(object_path, 'r') as f:
        objects = json.load(f)

    difficulties = []
    for obj in objects:
        diffs = obj.get('difficulties', [0] * len(obj.get('labels', [])))
        difficulties.append(torch.tensor(diffs, dtype=torch.int64, device=config.device))
    return difficulties


def evaluate(dataset, model, class_num, difficulties, iou_thresholds=None):
    model.eval()

    if iou_thresholds is None:
        iou_thresholds = [0.5]
    iou_thresholds = [float(t) for t in iou_thresholds]

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    for i in tqdm(range(len(dataset)), desc='Evaluating CenterNet mAP'):
        image_tensor, _, labels, original_image, original_boxes = dataset[i]

        pred_boxes, pred_labels, _, _, pred_scores = predict_box(model, image_tensor, original_image)

        if pred_boxes.numel() == 0:
            det_boxes.append(torch.empty((0, 4), dtype=torch.float32, device=config.device))
            det_labels.append(torch.empty((0,), dtype=torch.int64, device=config.device))
            det_scores.append(torch.empty((0,), dtype=torch.float32, device=config.device))
        else:
            det_boxes.append(pred_boxes.to(config.device))
            det_labels.append(pred_labels.to(config.device).long())
            det_scores.append(pred_scores.to(config.device).float())

        gt_boxes = torch.tensor(original_boxes, dtype=torch.float32, device=config.device)
        gt_labels = torch.tensor(labels, dtype=torch.int64, device=config.device)

        true_boxes.append(gt_boxes)
        true_labels.append(gt_labels)
        true_difficulties.append(difficulties[i] if i < len(difficulties) else torch.zeros(gt_labels.size(0), dtype=torch.int64, device=config.device))

    ap_by_iou = {}
    for thr in iou_thresholds:
        aps = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            true_difficulties,
            n_classes=class_num,
            device=config.device,
            iou_threshold=thr,
        )
        ap_by_iou[thr] = aps

    iou50 = min(iou_thresholds, key=lambda x: abs(x - 0.5))
    ap50 = ap_by_iou[iou50]
    map50 = ap50.mean().item()

    map_values = [ap.mean().item() for ap in ap_by_iou.values()]
    map50_95 = sum(map_values) / len(map_values)

    return {
        'mAP@0.50': map50,
        'mAP@0.50:0.95': map50_95,
        'AP@0.50': ap50,
        'AP_by_iou': ap_by_iou,
    }
