# 学習・推論に関するすべてのハイパーパラメータを管理する
import numpy as np
import torch
import os
import utils

# 使用するデータセット。「VOC2007」「VOC2007+2012」「fluid」から選択
target = "VOC2007"  # single-stage target: "VOC2007", "VOC2007+2012", "fluid"

# VOC / Fluid 各データセットのクラス名
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
fluid_labels = ('pump_erupt', 'shock', 'shock_half')

data_folder = utils.get_data_folder(target)

input_shape = [512, 512]   # 入力画像サイズ [H, W]
batch_size = 24             # バッチサイズ
loadworkers = 8             # DataLoader のワーカー数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用デバイス
num_classes = utils.get_num_classes(target)  # クラス数
learning_rate = 2.5e-4     # 初期学習率
weight_decay = 5e-4         # L2 正則化係数
epoch = 80                  # 単一ステージ学習のエポック数
test_epoch = 5              # 何エポックごとにテストするか
weight_save_path = "./weights"  # 重みの保存先
log_save_path = "./logs"        # ログの保存先
downsampled = 4             # ストライドの倍率（入力→出力特徴マップ）

# 転移学習の設定
enable_transfer_learning = False     # True: 2段階学習, False: 単一段階
pretrain_target = "VOC2007+2012"    # 事前学習データセット
finetune_target = "fluid"           # ファインチューニングデータセット
pretrain_epochs = 80                # 事前学習エポック数
finetune_epochs = 50                # ファインチューニングエポック数

