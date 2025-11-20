import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

class BiotTuner(LightningModule):
    def __init__(self, backbone, num_classes=10, lr=1e-4):
        super().__init__()
        self.backbone = backbone           # 预训练骨干
        # 把原模型最后一层替换成下游任务头
        # 以 ResNet50 为例，原 fc 是 2048→1000
        backbone.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )
        self.lr = lr
        # 1. 初始化两个 accuracy 计算器
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=num_classes)

    def forward(self, x):
        emb, pred_emb = self.backbone(x)
        return emb

    def training_step(self, batch, idx):
        x, y = batch
        logits = self(x)
        y = y.long()
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        logits = self(x)
        y = y.long()
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",  acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 只训练新 head，骨干先冻住
        return torch.optim.Adam(self.backbone.prediction.parameters(), lr=self.lr)