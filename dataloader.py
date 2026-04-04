import math
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import json, os, sys
import config
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors

def draw_gaussian(heatmap, center, radius, k=1):
    """ヒートマップに2Dガウス分布を描画する"""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    """2Dガウスカーネルを生成する"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    """IoU ≥ min_overlap を満たすガウス半径を計算する"""
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


class CenternetDataset(Dataset):
    def __init__(self, data_folder, input_shape, num_classes, dataset, mode):
        super(CenternetDataset, self).__init__()
        self.split = dataset.upper()
        assert self.split in {'TRAIN', 'TEST', 'TRAINVAL'}
        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        #self.annotation_lines   = annotation_lines
        self.length             = len(self.images)
        self.input_shape        = input_shape
        self.output_shape       = (int(input_shape[0]/config.downsampled) , int(input_shape[1]/config.downsampled))
        self.num_classes        = num_classes
        self.mode               = mode

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        images = self.images[index]
        objects = self.objects[index]
        boxes = objects["boxes"]
        labels = objects["labels"]
        labels = [x - 1 for x in labels]  

        #-------------------------------------------------#
        #   データ拡張を適用
        #-------------------------------------------------#
        if self.mode == "Train":
            images, boxes, labels = self.data_augment(images, boxes, labels, self.input_shape)
            boxes = boxes.numpy()
            labels = labels.numpy().tolist()

        elif self.mode == "Test":
            images, boxes = self.test_transforms(images, boxes, self.input_shape)
            boxes = boxes.numpy()

        elif self.mode == "Predict":
            images, boxes, original_images, original_boxes = self.val_transforms(images, boxes, self.input_shape)
            #images = v2.ToPILImage()(images)  # Convert to PIL Image
            boxes = boxes.numpy()
            return images, boxes, labels, original_images, original_boxes
                   
        batch_hm        = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask  = np.zeros((self.output_shape[0], self.output_shape[1], 1), dtype=np.float32)
             
        # 座標を出力特徴マップのスケールに変換（÷ダウンサンプル倍率）
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1) #w/4
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1) #h/4

        for i in range(boxes.shape[0]):
            bbox    = boxes[i].copy()
            cls_id  = int(labels[i])

            box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if box_h > 0 and box_w > 0:
                radius = gaussian_radius((math.ceil(box_h), math.ceil(box_w)))
                radius = max(0, int(radius))
                #-------------------------------------------------#
                #   正解ボックスに対応する特徴点（中心）を計算
                #-------------------------------------------------#
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  #center_x, center_y
                ct_int = ct.astype(np.int32)
                #print(ct, ct_int)
                #----------------------------#
                #   ガウスヒートマップを描画
                #----------------------------#
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                #---------------------------------------------------#
                #   幅・高さの正解値を設定
                #---------------------------------------------------#
                batch_wh[ct_int[1], ct_int[0], :] = 1. * box_w, 1. * box_h
                #---------------------------------------------------#
                #   中心点オフセットを計算
                #---------------------------------------------------#
                batch_reg[ct_int[1], ct_int[0], :] = ct - ct_int
                #---------------------------------------------------#
                #   対応位置のマスクを 1 に設定
                #---------------------------------------------------#
                batch_reg_mask[ct_int[1], ct_int[0], 0] = 1

        return images, batch_hm, batch_wh, batch_reg, batch_reg_mask


    def data_augment(self, image_path, boxes, labels, input_shape):
        transforms = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            v2.Resize(input_shape),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_orig = Image.open(image_path)
        H, W = img_orig.height, img_orig.width

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = boxes_tensor.reshape(0, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        if boxes_tensor.shape[0] > 0:
            boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]].clamp(0, W - 1)
            boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]].clamp(0, H - 1)

        boxes_tv = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(H, W))
        img_out, boxes_out = transforms(img_orig, boxes_tv)
        return img_out, boxes_out, labels_tensor


    def test_transforms(self, image_path, boxes, input_shape):
        img = Image.open(image_path)
        H = img.height
        W = img.width
        #origin_img = img.copy()
        transforms = v2.Compose([
            v2.ToImage(), 
            v2.Resize(input_shape),  # 入力サイズへリサイズ
            v2.ToDtype(torch.float32, scale=True),  # Tensor 化して [0, 1] に正規化
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))
        img, boxes = transforms(img, boxes)
        return img, boxes
    
    def val_transforms(self, image_path, boxes, input_shape):
        img = Image.open(image_path)
        H = img.height
        W = img.width
        origin_img = img.copy()
        origin_boxes = boxes.copy()
        transforms = v2.Compose([
            v2.ToImage(), 
            v2.Resize(input_shape),  # 入力サイズへリサイズ
            v2.ToDtype(torch.float32, scale=True),  # Tensor 化して [0, 1] に正規化
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))
        img, boxes = transforms(img, boxes)
        return img, boxes, origin_img, origin_boxes
    
        

# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    """バッチ内の各サンプルをスタックして Tensor に変換するコレート関数"""
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs            = torch.stack(imgs).float() if isinstance(imgs[0], torch.Tensor) else torch.from_numpy(np.array(imgs)).float()
    batch_hms       = torch.from_numpy(np.array(batch_hms)).float()
    batch_whs       = torch.from_numpy(np.array(batch_whs)).float()
    batch_regs      = torch.from_numpy(np.array(batch_regs)).float()
    batch_reg_masks = torch.from_numpy(np.array(batch_reg_masks)).float()
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks


if __name__ == '__main__':
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    data_folder = "Centernet/Dataset_VOC2007"
    input_shape = [512, 512]

    train_dataset = CenternetDataset(data_folder, input_shape, len(voc_labels), split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, 
                        collate_fn=centernet_dataset_collate, num_workers=4,
                        pin_memory=True, drop_last=True)
    for i, (images, hm_GT, wh_GT, offset_GT, reg_mask) in enumerate(train_loader):
        print(images.shape, hm_GT.shape, wh_GT.shape, offset_GT.shape, reg_mask.shape)
        sys.exit()

