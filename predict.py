from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from model import Centernet_model_ResNet50
import config, utils
import torch.nn.functional as F
import torchvision.ops as ops
import sys, os
from torch.utils.data import DataLoader
import torch
from dataloader import CenternetDataset, centernet_dataset_collate
import numpy as np

def pool_nms(hm_pred, kernel=3):
    """MaxPool で局所最大値のみを残す NMS（ヒートマップ用）"""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm_pred, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == hm_pred).float()  # 局所最大値のみを保持
    return hm_pred * keep  # ピーク以外の値を 0 にして抑制


def predict_box(model, images, original_images):
    """画像を推論してボックス・ラベル・スコアを返す"""
    images = images.unsqueeze(0).to(config.device)  
    H = original_images.height
    W = original_images.width

    hm, wh, offset = model(images)   
    hm = pool_nms(hm, kernel=3)  
    hm = hm.squeeze(0).permute(1, 2, 0).contiguous() 
    wh = wh.squeeze(0).permute(1, 2, 0).contiguous()
    offset = offset.squeeze(0).permute(1, 2, 0).contiguous()

    predicted_box_list = torch.empty(size=(0, 4), dtype=torch.float32)
    predicted_label_list = torch.empty(0, dtype=torch.int8)
    score_list = torch.empty(0, dtype=torch.float32)
    for i in range(hm.shape[-1]):
        scores, indices = torch.topk(hm[:, :, i].reshape(-1), k=50, dim=0)  
        for score, index in zip(scores, indices):
            score_value = score.item()
            index = index.item()
            if score_value > 0.3:
                max_x = index % hm.shape[1] 
                max_y = index // hm.shape[1]
                max_wh = wh[max_y, max_x, :]
                max_offset = offset[max_y, max_x, :]
                max_box = torch.tensor([max_x + max_offset[0], max_y + max_offset[1], abs(max_wh[0]), abs(max_wh[1])]).to("cpu")
                max_box = max_box.numpy()
                max_box = max_box * config.downsampled  # 特徴マップ座標 → 入力画像座標に変換
                
                predicted_box = torch.tensor([max_box[0] - max_box[2] / 2, max_box[1] - max_box[3] / 2, max_box[0] + max_box[2] / 2, max_box[1] + max_box[3] / 2], dtype=torch.float32)
                predicted_box = torch.tensor([
                    predicted_box[0] / config.input_shape[1] * W,
                    predicted_box[1] / config.input_shape[0] * H,
                    predicted_box[2] / config.input_shape[1] * W,
                    predicted_box[3] / config.input_shape[0] * H,
                ], dtype=torch.float32)
                predicted_box = torch.clip(predicted_box, min=torch.tensor([0,0,0,0]), max=torch.tensor([W, H, W, H]))  
                predicted_box = predicted_box.unsqueeze(0)
                predicted_label = torch.tensor([i], dtype=torch.int8)
                predicted_box_list = torch.cat((predicted_box_list, predicted_box), dim=0)
                predicted_label_list = torch.cat((predicted_label_list, predicted_label))
                score_list = torch.cat((score_list, torch.tensor([score_value], dtype=torch.float32)))

    if predicted_box_list.shape[0] != 0:
        keep = ops.nms(predicted_box_list.view(-1, 4), score_list.view(-1), iou_threshold=0.5)
        return predicted_box_list[keep], predicted_label_list[keep], score_list[keep]
    else:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int8),
            torch.empty((0,), dtype=torch.float32),
        )

            
def imagewithanchor(image, boxes, labels, number, label_names=None):
    """検出結果をPIL画像に描画して保存する"""
    if label_names is None:
        label_names = utils.get_labels(config.target)  # ターゲットに応じたラベル名を自動取得
    assert len(boxes) == len(labels), "Boxes and labels must have the same length"
    for box, label in zip(boxes, labels):
        label = label.item()  # Convert tensor to int
        box = box.numpy()  # Convert tensor to numpy array
        label_name = label_names[label]
        draw_box_label(image, box, label_name)
    image.save(plt_save_path+"/"+str(number)+".png")
        #image.show()
    image.close()


def draw_box_label(image, box, label_name):
    """矩形ボックス（赤）とラベルを描画する"""
    draw = ImageDraw.Draw(image)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=5)
    draw.text(box[:2], label_name, fill="Green")



if __name__ == '__main__':
    plt_save_path = "Result_img_test"
    weight_path = "Centernet/weights/fiuld_best.pth"

    # 学習済みモデルの重みを読み込む
    print(f"Loading weights from: {weight_path}")
    model = Centernet_model_ResNet50(num_classes=config.num_classes).to(config.device)
    model.load_state_dict(torch.load(weight_path, map_location=config.device, weights_only=True))
    model.eval()

    # 推論用データセットを作成
    dataset = CenternetDataset(config.data_folder, config.input_shape, len(config.voc_labels), "test", mode="Predict")

    for i in range(len(dataset)):
        images, boxes, labels, original_images, original_boxes = dataset[i]
        #print(images.shape, boxes, labels, original_images.size, original_boxes)
        predicted_box, predicted_label, predicted_score = predict_box(model, images, original_images)
        if predicted_box.shape[0] == 0:
            print(f"No valid prediction for image {i}.")
            continue
        else:
            imagewithanchor(original_images, predicted_box, predicted_label, number=i)
            print(f"scores: {predicted_score[:3].tolist()}")
            
        print(f"Image{i}/{len(dataset)}")
        
