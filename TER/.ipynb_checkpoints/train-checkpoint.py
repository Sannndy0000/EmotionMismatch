import torch

from torch.optim import AdamW
# from transformers import get_scheduler

from tqdm.auto import tqdm
from models import *

from utils import *
import numpy as np
import copy
import pandas as pd



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(train_loader, val_loader, label, checkpoint, lr, seed, num_epochs=10):
    model = BertMultiRegressionModel(checkpoint, seed, label)
    optimizer = AdamW(model.parameters(), lr)
    num_training_steps = num_epochs * len(train_loader)

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    best_ccc = 0
    for epoch in range(num_epochs):
        print("Epoch ", epoch, '-'*30)
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        print("Train Loss ", round(np.mean(losses), 4))

        model.eval()
        val_logits = []
        act_logits = []
        y_truth_valence = []
        y_truth_activation = []
        losses = []
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(loss.detach().cpu().numpy())
            val_logits.extend(list(outputs.logits[0].cpu().view(-1)))
            act_logits.extend(list(outputs.logits[1].cpu().view(-1)))
            y_truth_valence.extend(list(batch['valence'].cpu()))
            y_truth_activation.extend(list(batch['activation'].cpu()))
        print("Eval Loss ", round(np.mean(losses), 4))
        val_metric, act_metric = compute_metric(y_truth_valence, y_truth_activation, val_logits, act_logits)
        current_ccc = np.nanmean([val_metric['CCC'], act_metric['CCC']])
        if current_ccc > best_ccc:
            best_model = copy.deepcopy(model).cpu() 
            best_ccc = current_ccc
        
    return best_model



def test(model, test_loader, label):
    model = model.to(device)
    model.eval()
    val_logits = []
    act_logits = []
    y_truth_valence = []
    y_truth_activation = []
    subjects = []
    call_ids = []
    seg_ids = []
    losses = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # print(1)
        # print(list(batch.keys()))
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.detach().cpu().numpy())
        subjects += batch['sub_id'].cpu().tolist()
        call_ids += batch['call_id'].cpu().tolist()
        seg_ids += batch['seg_id'].cpu().tolist()
        val_logits.extend(outputs.logits[0].cpu().view(-1).tolist())
        act_logits.extend(outputs.logits[1].cpu().view(-1).tolist())
        y_truth_valence.extend(batch['valence'].cpu().tolist())
        y_truth_activation.extend(batch['activation'].cpu().tolist())
    print("Test Loss ", round(np.mean(losses), 4))
    val_metric, act_metric = compute_metric(y_truth_valence, y_truth_activation, val_logits, act_logits)
    
    results = pd.DataFrame({'subject': subjects, 'call_id':call_ids, 'seg_id':seg_ids, 'pred_valence': val_logits, 'pred_activation': act_logits, \
                                        'gt_valence': y_truth_valence, 'gt_activation':y_truth_activation})
    return results


