import torch
import torch.nn.functional as F
import torch.nn as nn
import config
import sys

class BoxLoss(nn.Module):
    """CenterNet の統合損失: focal_loss + 0.1×size_loss + offset_loss"""
    def __init__(self, alpha=2, beta=4, perimute=False):
        """
        Focal Loss for CenterNet
        :param alpha: Focusing parameter to reduce the impact of easy examples
        :param beta: Balancing parameter for positive and negative samples
        """
        super(BoxLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.perimute = perimute

    def forward(self, hm_pred, wh_pred, offset_pred, hm_GT, wh_GT, offset_GT, reg_mask_GT):
        if self.perimute:
            hm_GT = hm_GT.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            wh_GT = wh_GT.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            offset_GT = offset_GT.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            reg_mask_GT = reg_mask_GT.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Compute the losses
        focal_loss = self.Focal_loss(hm_pred, hm_GT, reg_mask_GT)
        size_loss = self.Size_loss(wh_pred, wh_GT, reg_mask_GT)
        offset_loss = self.Offset_loss(offset_pred, offset_GT, reg_mask_GT)
        
        #print("focal_loss:", focal_loss)
        #print("size_loss:", size_loss)
        #print("offset_loss:", offset_loss)

        # 各損失の重み付き和（size_loss は値域が大きいため 0.1 で正規化）
        lossALL = focal_loss*1.0 + size_loss*0.1 + offset_loss*1.0
        return focal_loss, size_loss, offset_loss, lossALL
    

    def Focal_loss(self, hm_pred, hm_GT, reg_mask_GT=None):
        """
        Compute the focal loss.
        :param hm_pred: Predicted heatmap (B, C, H, W)
        :param hm_GT: Ground truth heatmap (B, C, H, W)
        :param reg_mask_GT: not used; kept for API compatibility
        :return: Focal loss
        """
        eps = 1e-6  # Small value to avoid log(0)
        hm_pred = torch.clamp(hm_pred, min=eps, max=1-eps) # hm_pred を [eps, 1-eps] に制限

        #pos_mask = reg_mask_GT.eq(1).float()  # Indices where target == 1
        #neg_mask = reg_mask_GT.lt(1).float()  # Indices where target < 1
        # ガウスピーク領域（hm_GT >= 0.9）を正例、それ以外を負例とする
        pos_mask = hm_GT.ge(0.9).float()
        neg_mask = hm_GT.lt(0.9).float()

        # 正例損失: Focal Loss の正例項
        pos_loss = -torch.log(hm_pred) * torch.pow(1 - hm_pred, self.alpha) * pos_mask

        # 負例損失: ガウス重み付きで誤検出を抑制
        neg_loss = -torch.log(1 - hm_pred) * torch.pow(hm_pred, self.alpha) * torch.pow(1 - hm_GT, self.beta) * neg_mask

        # 正例数で正規化（ゼロ除算防止のため 1e-4 を加算）
        num_pos = pos_mask.sum()
        #neg_pos = neg_mask.sum()
        loss = (pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-4)

        return loss
    

    def Size_loss(self, wh_pred, wh_GT, reg_mask_GT):
        reg_mask_GT = reg_mask_GT.expand_as(wh_pred)
        loss = F.l1_loss(wh_pred, wh_GT, reduction="none") * reg_mask_GT
        num_pos = reg_mask_GT.sum()
        if num_pos == 0:
            return torch.tensor(0.0, device=wh_pred.device, dtype=wh_pred.dtype)
        return loss.sum()/num_pos
    

    def Offset_loss(self, offset_pred, offset_GT, reg_mask_GT):
        reg_mask_GT = reg_mask_GT.expand_as(offset_pred)
        loss = F.l1_loss(offset_pred, offset_GT, reduction="none") * reg_mask_GT
        num_pos = reg_mask_GT.sum()
        if num_pos == 0:
            return torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        return loss.sum() / num_pos



if __name__ == "__main__":
    # Simulated predictions and targets
    hmpred = torch.rand((1, 20, 128, 128), device=config.device)  # Predicted heatmap
    hmtarget = torch.zeros((1, 20, 128, 128), device=config.device)  # Ground truth heatmap
    hmtarget[:, 10, 64, 64] = 1  

    sizepred = torch.rand((1, 2, 128, 128), device=config.device)  
    sizeGT = torch.ones((1, 2, 128, 128), device=config.device)  
    offpred = torch.rand((1, 2, 128, 128), device=config.device)  
    offGT = torch.ones((1, 2, 128, 128), device=config.device)  
    reg_mask_GT = torch.zeros((1, 1, 128, 128), device=config.device)  
    reg_mask_GT[:, 0, 64, 64] = 1  

    # Initialize the focal loss
    focal_loss = BoxLoss(alpha=2, beta=4, perimute=False).cuda()

    # Compute the loss
    loss = focal_loss.forward(hmpred, sizepred, offpred, hmtarget, sizeGT, offGT, reg_mask_GT)
    print(loss)