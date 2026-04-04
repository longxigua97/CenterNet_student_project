import torch, sys
from torch.utils.data import DataLoader
import config
from dataloader import CenternetDataset, centernet_dataset_collate
import matplotlib.pyplot as plt
import numpy as np
import os , math
import cv2
import PIL.Image as Image
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors

#os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


def draw_gaussian(heatmap, center, radius, k=1):
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
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    """
    対象ボックスに対応するガウス半径を計算し、
    生成されるヒートマップが最小重なり条件を満たすようにする。

    引数:
        det_size: (height, width) ボックスサイズ
        min_overlap: 最小 IoU しきい値（通常 0.7）

    戻り値:
        radius: ガウスカーネル半径（整数）
    """
    height, width = det_size

    # case 1: ガウスとボックスの最小 IoU が min_overlap を満たす条件
    a1  = 1
    b1  = height + width
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    # case 2: ボックス基準でガウス被覆時の最小交差面積条件
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    # case 3: ガウスとボックスが重なる場合の境界条件
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def visualize_heatmap(image, heatmap):
    """元画像にヒートマップを重ねて表示する"""
    plt.imshow(image, cmap='gray', interpolation='nearest', alpha=0.8)       
    # 元画像にヒートマップを重ねる
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', alpha=0.5)
    #plt.title(f"Class {label[cls_idx]} Heatmap")
    plt.colorbar()
    #plt.savefig(f"{save_path}_class_{cls_idx}.png")
    plt.show()
    #plt.clf()  # 現在の描画をクリア

def data_augment(image_path, boxes, input_shape):
    """拡張後の画像とバウンディングボックスを返す（可視化確認用スタンドアローン版）"""
    img = Image.open(image_path)
    #origin_img = img.copy()
    H = img.height
    W = img.width
    #boxes = torch.tensor(boxes, dtype=torch.float32)
    transforms = v2.Compose([
        v2.ToImage(), 
        v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.6, 1.4)),  # ランダムアフィン: 平行移動 ±0.1、拡大縮小 0.6〜1.4、回転 ±10°
        v2.RandomHorizontalFlip(p=0.5),  # 50% の確率で水平反転
        v2.Resize(input_shape),  # 入力サイズへリサイズ
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 色調ゆらぎ
        v2.ToDtype(torch.float32, scale=True),  # Tensor 化して [0, 1] に正規化
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]),
    ])

    boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))
    img, boxes = transforms(img, boxes)
    return img, boxes

    
if __name__ == "__main__":
    img_path = "/home/shambhala/2D_Detection/centernet/test/000002.jpg"
    image = cv2.imread(img_path)
    image_w, image_h = image.shape[1], image.shape[0]
    print("image_w:", image_w)
    print("image_h:", image_h)
    
    xmin = 139
    ymin = 200
    xmax = 207
    ymax = 301

    # 元画像のヒートマップ
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    heatmap = np.zeros((image_h, image_w), dtype=np.float32)
    radius = gaussian_radius((math.ceil(ymax-ymin), math.ceil(xmax-xmin)))
    radius = max(0, int(radius))
    heatmap = draw_gaussian(heatmap, center, radius)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("radius:", radius)
    visualize_heatmap(image, heatmap)
    


    img_augmented, boxes_augmented = data_augment(img_path, [[xmin, ymin, xmax, ymax]], config.input_shape)
    img_augmented = v2.ToPILImage()(img_augmented)
    boxes_augmented = boxes_augmented.numpy()
    xmin_a = boxes_augmented[0][0]
    ymin_a = boxes_augmented[0][1]
    xmax_a = boxes_augmented[0][2]
    ymax_a = boxes_augmented[0][3]

    center_a = ((xmin_a + xmax_a) / 2, (ymin_a + ymax_a) / 2)
    heatmap_a = np.zeros((image_h, image_w), dtype=np.float32)
    radius_a = gaussian_radius((math.ceil(ymax_a-ymin_a), math.ceil(xmax_a-xmin_a)))
    radius_a = max(0, int(radius_a))
    heatmap_a = draw_gaussian(heatmap_a, center_a, radius_a)
    print("radius:", radius_a)
    visualize_heatmap(img_augmented, heatmap_a)
    
    #print("heatmap:", heatmap)
