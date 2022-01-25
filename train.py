import torch
import torch.nn as nn
from tracker.loss.loss import SiamRPNLoss
from tracker.model.model import SiamRPN
import torch.optim as optim

model = SiamRPN()
criterion = SiamRPNLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
cls_weight = 1.0
loc_weight = 1.2


epochs = 1
for epoch in range(epochs):
    loss_total_val = 0
    loss_loc_val = 0
    loss_cls_val = 0
    for i, data in enumerate(train_loader):
        template, search, label_cls, label_loc, label_loc_weight = data
        loss = model(template, search)

        loss_total_val += loss["cls_loss"]
        loss_loc_val = += loss["cls_loss"]
        loss_cls_val = += loss["cls_loss"]