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
            input_ = input_["output_sigmoid"]
            target = target.argmax(1).long()
        elif self.loss_type == "bce":
            input_ = input_["output_sigmoid"]
            target = target.float()

        return self.loss(input_, target)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
