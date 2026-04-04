# CenterNet 学習スクリプト
# 単一ステージ学習 / 2段階転移学習（VOC事前学習 → Fluidファインチューン）の両方に対応
from dataloader import CenternetDataset, centernet_dataset_collate
from model import Centernet_model_ResNet50
from loss import BoxLoss
import mAP
import config, utils
import torch
from torch.utils.data import DataLoader
import time, os
from tqdm import tqdm
import torch.optim as optim
import pandas as pd


def resolve_split(data_folder, candidates):
    """Pick the first existing split by checking <SPLIT>_images/objects.json."""
    for split in candidates:
        image_json = os.path.join(data_folder, f'{split.upper()}_images.json')
        object_json = os.path.join(data_folder, f'{split.upper()}_objects.json')
        if os.path.isfile(image_json) and os.path.isfile(object_json):
            return split
    raise FileNotFoundError(
        f"No valid split found in {data_folder} for candidates: {candidates}"
    )

def train(train_loader, model, lossfunction, optimizer, lr_scheduler, log_df, epoch, total_epochs, stage):
    model.train()
    train_loss = 0
    focal_loss_mean = 0
    size_loss_mean = 0
    offset_loss_mean = 0
    start_time = time.time()

    for i, (images, hm_GT, wh_GT, offset_GT, reg_mask) in enumerate(train_loader):
        images = images.to(config.device)
        hm_GT = hm_GT.to(config.device)
        wh_GT = wh_GT.to(config.device)
        offset_GT = offset_GT.to(config.device)
        reg_mask = reg_mask.to(config.device)

        #print(images.shape, hm_GT.shape, wh_GT.shape, offset_GT.shape, reg_mask.shape)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward prop.
        hm_pred, wh_pred, offset_pred = model(images)
        #print(hm_pred.shape, wh_pred.shape, offset_pred.shape)
        
        # Loss
        focal_loss, size_loss, offset_loss, lossALL = lossfunction(hm_pred, wh_pred, offset_pred, hm_GT, wh_GT, offset_GT, reg_mask)  
        
        #    print(epoch, i, loss.item(), f"iter_time: {time.time() - iter_start:.2f}s")
        train_loss += lossALL.item() / len(train_loader)
        focal_loss_mean += focal_loss.item() / len(train_loader)
        size_loss_mean += size_loss.item() / len(train_loader)
        offset_loss_mean += offset_loss.item() / len(train_loader)
        
        # Backward prop.
        lossALL.backward()

        # Update model
        optimizer.step()

        del images, hm_GT, wh_GT, offset_GT, reg_mask, hm_pred, wh_pred, offset_pred
    lr_scheduler.step()
    log_df.loc[len(log_df)] = {'stage': stage, 'epoch': epoch, 'mode': 'train', 'ALLloss': train_loss, 'focal_loss': focal_loss_mean, 'size_loss': size_loss_mean,
            'offset_loss': offset_loss_mean, 'learning_rate': optimizer.param_groups[0]['lr']}
    
    epoch_time = time.time() - start_time
    return train_loss, focal_loss_mean, size_loss_mean, offset_loss_mean, optimizer.param_groups[0]['lr'], epoch_time

def test(test_loader, model, lossfunction, log_df, epoch, stage):
    """テストセットで損失を評価してログに記録する"""
    model.eval()
    test_loss = 0
    focal_loss_mean = 0
    size_loss_mean = 0
    offset_loss_mean = 0
    with torch.no_grad():
        for i, (images, hm_GT, wh_GT, offset_GT, reg_mask) in enumerate(test_loader):
            images = images.to(config.device)
            hm_GT = hm_GT.to(config.device)
            wh_GT = wh_GT.to(config.device)
            offset_GT = offset_GT.to(config.device)
            reg_mask = reg_mask.to(config.device)

            # Forward prop.
            hm_pred, wh_pred, offset_pred = model(images)

            # Loss
            focal_loss, size_loss, offset_loss, lossALL = lossfunction(hm_pred, wh_pred, offset_pred, hm_GT, wh_GT, offset_GT, reg_mask)  
            test_loss += lossALL.item() / len(test_loader)
            focal_loss_mean += focal_loss.item() / len(test_loader)
            size_loss_mean += size_loss.item() / len(test_loader)
            offset_loss_mean += offset_loss.item() / len(test_loader)
            del images, hm_GT, wh_GT, offset_GT, reg_mask, hm_pred, wh_pred, offset_pred
    
    log_df.loc[len(log_df)] = {'stage': stage, 'epoch': epoch, 'mode': 'test', 'ALLloss': test_loss, 'focal_loss': focal_loss_mean, 'size_loss': size_loss_mean,
            'offset_loss': offset_loss_mean, 'learning_rate': '-'}
    return test_loss, focal_loss_mean, size_loss_mean, offset_loss_mean


def create_dataloaders(target_name):
    """指定データセット用の DataLoader を作成して返す"""
    labels = utils.get_labels(target_name)
    data_folder = utils.get_data_folder(target_name)
    train_split = resolve_split(data_folder, ["train", "trainval"])
    test_split = resolve_split(data_folder, ["test", "trainval", "train"])

    train_dataset = CenternetDataset(data_folder, config.input_shape, len(labels), train_split, mode="Train")
    test_dataset = CenternetDataset(data_folder, config.input_shape, len(labels), test_split, mode="Test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=centernet_dataset_collate,
        num_workers=config.loadworkers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=centernet_dataset_collate,
        num_workers=config.loadworkers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader, len(labels)


def load_partial_weights(model, weight_path):
    """形状が一致するパラメータのみを選択的にロードする（転移学習用）"""
    if not os.path.isfile(weight_path):
        print(f"Skip loading pretrained weights. File not found: {weight_path}")
        return

    try:
        checkpoint = torch.load(weight_path, map_location=config.device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location=config.device)

    model_dict = model.state_dict()
    filtered = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered)}/{len(model_dict)} compatible parameters from {weight_path}")


def train_stage(stage_name, target_name, epochs, init_weights=None, lr_scale=1.0):
    """1ステージ分の学習を実行して最終重みパスとログを返す
    lr_scale: 転移学習時は 0.1 を指定して学習率を下げる
    """
    print(f"\n========== Start stage: {stage_name} | target: {target_name} ==========")
    train_loader, test_loader, num_classes = create_dataloaders(target_name)

    model = Centernet_model_ResNet50(num_classes=num_classes).to(config.device)
    if init_weights is not None:
        load_partial_weights(model, init_weights)

    loss = BoxLoss(alpha=2, beta=4, perimute=True).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate * lr_scale, weight_decay=config.weight_decay)

    if epochs >= 3:
        milestones = sorted(set([max(1, int(epochs * 0.75)), max(1, int(epochs * 0.9))]))
    else:
        milestones = [1]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    stage_log_df = pd.DataFrame(columns=[
        'stage',
        'epoch',
        'mode',
        'ALLloss',
        'focal_loss',
        'size_loss',
        'offset_loss',
        'learning_rate',
    ])

    os.makedirs(config.weight_save_path, exist_ok=True)
    best_test_loss = float('inf')
    for epoch in tqdm(range(epochs), desc=f'{stage_name}'):
        train_all_loss, train_focal_loss, train_size_loss, train_offset_loss, current_lr, epoch_time = train(
            train_loader,
            model,
            loss,
            optimizer,
            lr_scheduler,
            stage_log_df,
            epoch=epoch,
            total_epochs=epochs,
            stage=stage_name,
        )

        tqdm.write(
            f"Stage [{stage_name}] Epoch [{epoch+1}/{epochs}] Train - "
            f"ALL: {train_all_loss:.6f}, Focal: {train_focal_loss:.6f}, "
            f"Size: {train_size_loss:.6f}, Offset: {train_offset_loss:.6f}, "
            f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s"
        )

        if epoch % config.test_epoch == 0:
            test_all_loss, test_focal_loss, test_size_loss, test_offset_loss = test(test_loader, model, loss, stage_log_df, epoch, stage=stage_name)
            tqdm.write(
                f"Stage [{stage_name}] Epoch [{epoch+1}/{epochs}] Test  - "
                f"ALL: {test_all_loss:.6f}, Focal: {test_focal_loss:.6f}, "
                f"Size: {test_size_loss:.6f}, Offset: {test_offset_loss:.6f}"
            )
            torch.save(model.state_dict(), os.path.join(config.weight_save_path, f'{stage_name}_epoch_{epoch}.pth'))
            if test_all_loss < best_test_loss:
                best_test_loss = test_all_loss
                torch.save(model.state_dict(), os.path.join(config.weight_save_path, f'{stage_name}_best.pth'))
                tqdm.write(f"  -> New best test loss: {best_test_loss:.6f}, saved best model.")

    final_stage_weight = os.path.join(config.weight_save_path, f'{stage_name}_final.pth')
    torch.save(model.state_dict(), final_stage_weight)
    print(f"Saved final stage weight: {final_stage_weight}")
    return final_stage_weight, stage_log_df


if __name__ == "__main__":

    # GPU 利用可否を確認
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU for training.", config.device)
    else:
        print("CUDA is available. Using GPU for training.", config.device)
        print("Number of GPUs available:", torch.cuda.device_count())

    os.makedirs(config.log_save_path, exist_ok=True)
    all_logs = []

    # enable_transfer_learning を有効にする場合は、VOC で事前学習 → Fluid でファインチューンの2段階学習を実行
    if config.enable_transfer_learning:
        pretrain_weight, pretrain_log = train_stage(
            stage_name='pretrain_voc',
            target_name=config.pretrain_target,
            epochs=config.pretrain_epochs,
            init_weights=None,
        )
        all_logs.append(pretrain_log)

        ## 転移学習では小さめの学習率を使用してファインチューンを実行
        finetune_weight, finetune_log = train_stage(
            stage_name='finetune_fluid',
            target_name=config.finetune_target,
            epochs=config.finetune_epochs,
            init_weights=pretrain_weight,
            lr_scale=0.1,  
        )
        all_logs.append(finetune_log)
        print(f"Transfer learning done. Final fluid model: {finetune_weight}")
    else:
    # 単一ステージ学習を実行
        single_stage_weight, single_stage_log = train_stage(
            stage_name=f"single_{config.target.lower()}",
            target_name=config.target,
            epochs=config.epoch,
            init_weights=None,
        )
        all_logs.append(single_stage_log)
        print(f"Single-stage training done. Final model: {single_stage_weight}")

    # logをCSVに保存
    merged_log = pd.concat(all_logs, ignore_index=True)
    logging_file = os.path.join(config.log_save_path, 'log.csv')
    merged_log.to_csv(logging_file, index=False, mode='w', header=True)
    print(f"Training logs saved to: {logging_file}")

    # mAP評価
    eval_target = config.finetune_target if config.enable_transfer_learning else config.target
    eval_labels = utils.get_labels(eval_target)
    eval_data_folder = utils.get_data_folder(eval_target)
    eval_split = mAP.find_eval_split(eval_data_folder)

    eval_dataset = CenternetDataset(
        data_folder=eval_data_folder,
        input_shape=config.input_shape,
        num_classes=len(eval_labels),
        dataset=eval_split,
        mode='Predict',
    )
    eval_difficulties = mAP.load_difficulties(eval_data_folder, eval_split)

    final_model = Centernet_model_ResNet50(num_classes=len(eval_labels)).to(config.device)
    final_weight_path = finetune_weight if config.enable_transfer_learning else single_stage_weight
    try:
        final_state = torch.load(final_weight_path, map_location=config.device, weights_only=True)
    except TypeError:
        final_state = torch.load(final_weight_path, map_location=config.device)
    final_model.load_state_dict(final_state)
    final_model.eval()

    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    metrics = mAP.evaluate(
        eval_dataset,
        final_model,
        len(eval_labels),
        eval_difficulties,
        iou_thresholds=iou_thresholds,
    )
    print('\nmAP@0.50: {:.3f}'.format(metrics['mAP@0.50']))
    print('mAP@0.50:0.95: {:.3f}'.format(metrics['mAP@0.50:0.95']))
    print('\nPer-class AP:')
    for i, ap in enumerate(metrics['AP@0.50']):
        print('{}: {:.3f}'.format(eval_labels[i], ap.item()))

