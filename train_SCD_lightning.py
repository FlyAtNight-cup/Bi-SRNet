import argparse
from argparse import Namespace
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import WandbLogger
from models.SSCDl import SSCDl as Net
from datasets import RS_ST as RS
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.loss import CrossEntropyLoss2d, ChangeSimilarity, weighted_BCE_logits
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from tqdm import tqdm
from typing import Any, Dict, Tuple
import wandb
import os
import lightning

def adjust_lr(optimizer: optim.Optimizer, 
              curr_iter: float, 
              all_iter: float, 
              init_lr: float, 
              lr_decay_power: float):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** lr_decay_power)
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr

class BiSRNet(lightning.LightningModule):
    def __init__(self, net: nn.Module, opt: Namespace, len_train_loader: int):
        super().__init__()
        self.net = net
        self.opt = opt
        self.len_train_loader = len_train_loader
        self.criterion = CrossEntropyLoss2d(ignore_index=0)
        self.criterion_sc = ChangeSimilarity()
        self.bestaccT = 0
        self.bestFscdV = 0.0
        self.bestloss = 1.0
    
    def forward(self, imgs_A: torch.Tensor, imgs_B: torch.Tensor) -> Tuple[torch.Tensor,  torch.Tensor, torch.Tensor]:
        return self.net(imgs_A, imgs_B)
    
    def on_fit_start(self) -> None:
        self.all_iters = float(self.len_train_loader) * self.opt.epoch
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)]
    
    def on_train_epoch_start(self) -> None:
        self.acc_meter = AverageMeter()
        self.train_seg_loss = AverageMeter()
        self.train_bn_loss = AverageMeter()
        self.train_sc_loss = AverageMeter()
        
        self.curr_iter = self.current_epoch * self.len_train_loader
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        running_iter = self.curr_iter + batch_idx + 1
        adjust_lr(self.optimizers(), running_iter, self.all_iters, self.opt.lr, self.opt.lr_decay_power)
        imgs_A, imgs_B, labels_A, labels_B = batch
        imgs_A: torch.Tensor = imgs_A.float()
        imgs_B: torch.Tensor = imgs_B.float()
        labels_bn: torch.Tensor = (labels_A>0).unsqueeze(1).float()
        labels_A: torch.Tensor = labels_A.long()
        labels_B: torch.Tensor = labels_B.long()
        out_change, outputs_A, outputs_B = self(imgs_A, imgs_B)
        out_change: torch.Tensor
        outputs_A: torch.Tensor
        outputs_B: torch.Tensor
        assert outputs_A.size()[1] == RS.num_classes
        loss_seg: torch.Tensor = self.criterion(outputs_A, labels_A) * 0.5 +  self.criterion(outputs_B, labels_B) * 0.5     
        loss_bn: torch.Tensor = weighted_BCE_logits(out_change, labels_bn)
        loss_sc: torch.Tensor = self.criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
        loss: torch.Tensor = loss_seg + loss_bn + loss_sc
        
        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = torch.sigmoid(out_change).cpu().detach()>0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long()).numpy()
        preds_B = (preds_B*change_mask.squeeze().long()).numpy()
        acc_curr_meter = AverageMeter()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_curr_meter.update(acc)
        self.acc_meter.update(acc_curr_meter.avg)
        self.train_seg_loss.update(loss_seg.cpu().detach().numpy())
        self.train_bn_loss.update(loss_bn.cpu().detach().numpy())
        self.train_sc_loss.update(loss_sc.cpu().detach().numpy())

        self.log("train/seg_loss", float(self.train_seg_loss.val), sync_dist=True)
        self.log("train/sc_loss", float(self.train_sc_loss.val), sync_dist=True)
        self.log("train/accuracy", float(self.acc_meter.val), sync_dist=True)

        
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().param_groups[0]['lr'], sync_dist=True)
        if self.acc_meter.avg > self.bestaccT: 
            self.bestaccT = self.acc_meter.avg
    
    def on_validation_epoch_start(self) -> None:
        self.val_loss = AverageMeter()
        self.acc_meter = AverageMeter()
        self.preds_all = []
        self.labels_all = []
        self.pred = False
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        imgs_A, imgs_B, labels_A, labels_B = batch
        imgs_A: torch.Tensor = imgs_A.float()
        imgs_B: torch.Tensor = imgs_B.float()
        labels_A: torch.Tensor = labels_A.long()
        labels_B: torch.Tensor = labels_B.long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = self(imgs_A, imgs_B)
            loss_A: torch.Tensor = self.criterion(outputs_A, labels_A)
            loss_B: torch.Tensor = self.criterion(outputs_B, labels_B)
            loss: torch.Tensor = loss_A * 0.5 + loss_B * 0.5
        self.val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A: torch.Tensor = outputs_A.cpu().detach()
        outputs_B: torch.Tensor = outputs_B.cpu().detach()
        change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long()).numpy()
        preds_B = (preds_B*change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            self.preds_all.append(pred_A)
            self.preds_all.append(pred_B)
            self.labels_all.append(label_A)
            self.labels_all.append(label_B)
            acc = (acc_A + acc_B)*0.5
            self.acc_meter.update(acc)

        if self.pred:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            self.log("pred/image_A", wandb.Image(pred_A_color, caption=f'BiSR-Net_A_{self.current_epoch}.png'), sync_dist=True)
            self.log("pred/image_B", wandb.Image(pred_B_color, caption=f'BiSR-Net_B_{self.current_epoch}.png'), sync_dist=True)
            self.pred = False
    
    
    def on_validation_epoch_end(self) -> None:
        Fscd, IoU_mean, Sek = SCDD_eval_all(self.preds_all, self.labels_all, RS.num_classes)

        self.log("val/loss", self.val_loss.average(), sync_dist=True)
        self.log("val/Fscd", Fscd, sync_dist=True)
        self.log("val/Accuracy", self.acc_meter.average(), sync_dist=True)

        
        if Fscd > self.bestFscdV:
            self.bestFscdV = Fscd
            self.bestaccV = self.acc_meter.average()
            self.bestloss = self.val_loss.average()
            torch.save(net.state_dict(), os.path.join(self.opt.checkpoint_path, 'BiSR-Net_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth'\
                %(self.current_epoch, IoU_mean * 100, Sek * 100, Fscd * 100, self.acc_meter.average() * 100)) )


def train(net: nn.Module, 
          dataloaders: Dict[str, DataLoader], 
          criterion: nn.Module, 
          optimizer: optim.Optimizer,
          scheduler: object,
          opt: Namespace):
    bestaccT = 0
    bestFscdV = 0.0
    bestloss = 1.0
    wandb.init(name=opt.exp_name + "_V" + str(opt.exp_version), project='BiSR-Net', config=opt.__dict__)
    
    criterion_sc = ChangeSimilarity().cuda()
    all_iters = float(len(dataloaders['train']) * opt.epoch)
    for current_epoch in range(opt.epoch):
        torch.cuda.empty_cache()
        net.train()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()
        
        curr_iter = current_epoch * len(train_loader)
        pbar = tqdm(dataloaders['train'])
        for i, data in enumerate(pbar):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, opt.lr, opt.lr_decay_power)
            imgs_A, imgs_B, labels_A, labels_B = data
            imgs_A: torch.Tensor = imgs_A.float().cuda()
            imgs_B: torch.Tensor = imgs_B.float().cuda()
            labels_bn: torch.Tensor = (labels_A>0).unsqueeze(1).float().cuda()
            labels_A: torch.Tensor = labels_A.long().cuda()
            labels_B: torch.Tensor = labels_B.long().cuda()
            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            out_change: torch.Tensor
            outputs_A: torch.Tensor
            outputs_B: torch.Tensor
            assert outputs_A.size()[1] == RS.num_classes
            loss_seg: torch.Tensor = criterion(outputs_A, labels_A) * 0.5 +  criterion(outputs_B, labels_B) * 0.5     
            loss_bn: torch.Tensor = weighted_BCE_logits(out_change, labels_bn)
            loss_sc: torch.Tensor = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
            loss: torch.Tensor = loss_seg + loss_bn + loss_sc
            loss.backward()
            optimizer.step()
            
            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = torch.sigmoid(out_change).cpu().detach()>0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).numpy()
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B)*0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            pbar.set_description_str('[epoch %d]' % (current_epoch)) 
            pbar.set_postfix_str('[acc %.2f]' % (acc_meter.val * 100))
            wandb.log({
                "train/seg_loss": train_seg_loss.val,
                "train/sc_loss": train_sc_loss.val,
                "train/accuracy": acc_meter.val
            })
        wandb.log({
            "lr": optimizer.param_groups[0]['lr']
        })
        pbar.close()
        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(net, dataloaders['val'], criterion, current_epoch)
        if acc_meter.avg > bestaccT: 
            bestaccT = acc_meter.avg
        if Fscd_v > bestFscdV:
            bestFscdV = Fscd_v
            bestaccV = acc_v
            bestloss = loss_v
            torch.save(net.state_dict(), os.path.join(opt.checkpoint_path, 'BiSR-Net_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth'\
                %(current_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100)) )
    print("%.2f %.2f %.2f\n" % (bestFscdV, bestaccV, bestloss))


def validate(net: nn.Module, val_loader: DataLoader, criterion: nn.Module, curr_epoch: int):
    net.eval()
    torch.cuda.empty_cache()
    val_loss = AverageMeter()
    acc_meter = AverageMeter()
    
    preds_all = []
    labels_all = []
    pbar = tqdm(val_loader)
    for vi, data in enumerate(pbar):
        imgs_A, imgs_B, labels_A, labels_B = data
        imgs_A: torch.Tensor = imgs_A.float().cuda()
        imgs_B: torch.Tensor = imgs_B.float().cuda()
        labels_A: torch.Tensor = labels_A.long().cuda()
        labels_B: torch.Tensor = labels_B.long().cuda()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A: torch.Tensor = criterion(outputs_A, labels_A)
            loss_B: torch.Tensor = criterion(outputs_B, labels_B)
            loss: torch.Tensor = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A: torch.Tensor = outputs_A.cpu().detach()
        outputs_B: torch.Tensor = outputs_B.cpu().detach()
        change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long()).numpy()
        preds_B = (preds_B*change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B)*0.5
            acc_meter.update(acc)

        if vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            wandb.log({
                "pred/image_A": wandb.Image(pred_A_color, caption=f'BiSR-Net_A_{curr_epoch}.png'),
                "pred/image_B": wandb.Image(pred_B_color, caption=f'BiSR-Net_B_{curr_epoch}.png')
            })
    
    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)
    
    pbar.close()
    
    print('Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(val_loss.average(), Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))

    wandb.log({
        "val/loss": val_loss.average(),
        "val/Fscd": Fscd,
        "val/Accuracy": acc_meter.average()
    })

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='the path of dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int,default=8)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_decay_power', type=float, default=1.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--checkpoint_path', default='./checkpoints')
    parser.add_argument('--exp_name', default='exp')
    parser.add_argument('--exp_version', default=1)
    parser.add_argument('--offline', default=False)
    parser.add_argument('--gpus', default=[0, 1, 2])
    args = parser.parse_args()
    
    train_loader = DataLoader(RS.Data(args.data_dir, 'train', random_flip=True), 
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers, 
                              shuffle=True)
    val_loader = DataLoader(RS.Data(args.data_dir, 'val'), 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            shuffle=False)
    net = BiSRNet(Net(3, num_classes=RS.num_classes), args, len(train_loader))
    logger = WandbLogger(name=args.exp_name + "_V" + str(args.exp_version), 
                         project='BiSR-Net', 
                         config=args.__dict__,
                         offline=args.offline)
    trainer = lightning.Trainer(
        devices=args.gpus,
        logger=logger,
        max_epochs=args.epoch
    )
    
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
    
    # train(net, {'train': train_loader, 'val': val_loader}, criterion, optimizer, scheduler, args)
    