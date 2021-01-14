from torch import nn


class ResNetLoss(nn.Module):
    def __init__(self, loss_type="ce"):
        super().__init__()

        self.loss_type = loss_type
        if loss_type == "ce":
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == "bce":
            self.loss = nn.BCELoss()

    def forward(self, input_, target):
        if self.loss_type == "ce":
            input_ = input_["multiclass_proba"]
            target = target.argmax(1).long()
        elif self.loss_type == "bce":
            input_ = input_["multilabel_proba"]
            target = target.float()

        return self.loss(input_, target)
