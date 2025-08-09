import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from tools.callbacks import ModelAveragingCallback
#os.environ['CUDA_VISIBLE_DEVICES']="1"
#os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # 避免NCCL问题


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

class Counter:
    def __init__(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return

    def update(self, value, num_updata=1):
        self.count += num_updata
        self.sum += value * num_updata
        self.avg = self.sum / self.count
        return

    def clear(self):
        self.count, self.sum, self.avg = 0, 0, 0
        return

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x, mask=None):

        seg_pre = self.net(x, mask)
        return seg_pre

    # def training_step(self, batch, batch_idx):
    #     img, mask = batch['img'], batch['gt_semantic_seg']
    #
    #     prediction = self.net(img)
    #     loss = self.loss(prediction, mask)
    #
    #     if self.config.use_aux_loss:
    #         pre_mask = nn.Softmax(dim=1)(prediction[0])
    #     else:
    #         pre_mask = nn.Softmax(dim=1)(prediction)
    #
    #     pre_mask = pre_mask.argmax(dim=1)
    #     for i in range(mask.shape[0]):
    #         self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
    #     #print('loss:', loss)
    #
    #
    #     return {"loss": loss}

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        # print("img",img.shape)
        # print("mask",mask.shape)

        prediction, contrast_logits, contrast_target = self.net(img, mask)


        total_loss = 0  # 初始化总损失

        # 遍历所有阶段的输出
        for pred, logits, target in zip(prediction, contrast_logits, contrast_target):
        # for pred, logits in zip(prediction, contrast_logits):
            # 计算当前阶段的损失
            loss_stage = self.loss(
                pred,  # segmentation output for this stage
                mask,  # ground truth
                contrast_logits=logits,  # contrastive logits for this stage
                contrast_target=target  # contrastive targets for this stage
            )
            total_loss += loss_stage  # 累积每个阶段的损失


        if self.config.use_aux_loss:
            # pre_mask = nn.Softmax(dim=1)(prediction[0])
            pre_mask = nn.Softmax(dim=1)(prediction[3])
        else:
           # pre_mask = nn.Softmax(dim=1)(prediction[0])
            pre_mask = nn.Softmax(dim=1)(prediction[3])

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        #print('loss:', loss)
        #loss = 0.6 * loss1 + 0.6 * loss2 + 0.6 * loss3 + loss4


        return {"loss": total_loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA
                      }
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction, _, _= self.forward(img, mask)
        pre_mask = nn.Softmax(dim=1)(prediction[3])
        # pre_mask = nn.Softmax(dim=1)(prediction[0])
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val1 = self.loss(prediction[0], mask)
        loss_val2 = self.loss(prediction[1], mask)
        loss_val3 = self.loss(prediction[2], mask)
        loss_val4 = self.loss(prediction[3], mask)
        loss_val = loss_val1 + loss_val2 + loss_val3 + loss_val4
        # #loss_val = 0.6 * loss_val1 + 0.6 * loss_val2 + 0.6 * loss_val3 + loss_val4
        # loss_val = (self.loss_weights[0] * loss_val1 +
        #             self.loss_weights[1] * loss_val2 +
        #             self.loss_weights[2] * loss_val3 +
        #             self.loss_weights[3] * loss_val4)

        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():

    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    # # 初始化 ModelAveragingCallback
    # if config.use_model_averaging:
    #     model_averaging_callback = ModelAveragingCallback(
    #         device=config.device,
    #         avg_fn=config.avg_fn
    #     )
    # else:
    #     model_averaging_callback = None

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    #     # 设置回调列表，如果启用了模型加权平均回调，添加到训练器中
    # callbacks = [checkpoint_callback]
    # if model_averaging_callback:
    #     callbacks.append(model_averaging_callback)
    #

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='ddp_find_unused_parameters_true', #strategy='auto',
                         logger=logger)
                         # enable_progress_bar=True, logger=False)
                   
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
