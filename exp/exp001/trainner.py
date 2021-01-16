from ipdb import set_trace as st
from loguru import logger
import configuration as C
from criterion import mixup_criterion
from utils import mixup_data


# def train_fold(self, i_fold: int) -> Tuple[
        # Model, np.array, np.array, float]:

def train_fold(i_fold, trn_tp, config):
    logger.info(':: in ::')
        # model, device, train_loader, optimizer,
        # scheduler, loss_func, mixup=False):
    mixup = config['globals']['mixup']
    device = config['globals']['device']
    trn_idxs, val_idxs = C.get_index_fold(trn_tp, i_fold, config)
    trn_tp_trn = trn_tp.iloc[trn_idxs].reset_index(drop=True)
    trn_tp_val = trn_tp.iloc[val_idxs].reset_index(drop=True)
    trn_loader = C.get_trn_val_loader(trn_tp_trn, 'train', config)
    val_loader = C.get_trn_val_loader(trn_tp_val, 'valid', config)

    model = C.get_model(config).to(device)
    criterion = C.get_criterion(config)
    optimizer = C.get_optimizer(model, config)
    scheduler = C.get_scheduler(optimizer, config)

    # st()
    model.train()
    epoch_train_loss = 0
    for batch_idx, (data, target) in enumerate(trn_loader):
        logger.info(f'{batch_idx + 1}/{len(trn_loader)}')
        data, target = data.to(device), target.to(device)
        if mixup:
            data, targets_a, targets_b, lam = mixup_data(data, target, alpha=1.0)
    
        optimizer.zero_grad()
        output = model(data)
        if mixup:
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()*data.size(0)
    scheduler.step()
    loss = epoch_train_loss / len(trn_loader.dataset)
    del data
    logger.info(':: out ::')
    return loss
