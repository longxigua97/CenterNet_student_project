# CenterNet モデル定義
# ResNet-50 エンコーダ + 転置畳み込みデコーダ + 3ヘッド（ヒートマップ / 幅高さ / オフセット）
import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import config

class resnet50_Decoder(nn.Module):
    """転置畳み込みによる3段階アップサンプリングデコーダ
    16×16×2048 → 128×128×64
    """
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False
        
        #----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   ConvTranspose2d でアップサンプリングを行う。
        #   各段で特徴マップの縦横サイズを 2 倍にする。
        #----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class resnet50_Head(nn.Module):
    """3種類の予測ヘッド: ヒートマップ / 幅高さ / 中心点オフセット"""
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1, dropout=0.2):
        super(resnet50_Head, self).__init__()
        #-----------------------------------------------------------------#
        #   抽出特徴に対して分類予測と回帰予測を行う
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # ヒートマップ予測ヘッド
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 幅・高さ予測ヘッド
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点オフセット予測ヘッド
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channel, 2,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid()   # ヒートマップ（0〜1 に正規化）
        wh = self.wh_head(x)              # 幅・高さ予測
        offset = self.reg_head(x)         # 中心点の量子化誤差補正
        return hm, wh, offset


class Centernet_model_ResNet50(nn.Module):
    """CenterNet フルモデル: Encoder + Decoder + Head"""
    def __init__(self, num_classes=80, weight=ResNet50_Weights.DEFAULT):
        super(Centernet_model_ResNet50, self).__init__()
        # ResNet-50 を読み込み、avgpool と fc を除いて layer4 まで使用
        Encoder = models.resnet50(weights=weight)
        self.Encoder = nn.Sequential(*list(Encoder.children())[:-2])
        self.Decoder = resnet50_Decoder(2048)
        self.Head = resnet50_Head(num_classes=num_classes)

    def forward(self, x):
        x = self.Encoder(x)              # ResNet-50 特徴抽出
        x = self.Decoder(x)              # アップサンプリング
        hm, wh, offset = self.Head(x)    # 各ヘッドの予測
        return hm, wh, offset 

    
if __name__ == '__main__':
    print(config.device)
    print(torch.cuda.is_available())
    img = torch.ones(32, 3, 512, 512).to(config.device)
    model = Centernet_model_ResNet50(num_classes=2).to(config.device)
    output = model.forward(img)
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
    print(output[0].shape, output[1].shape, output[2].shape)

    


