import torch
import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score,r2_score,matthews_corrcoef,confusion_matrix,average_precision_score
def evaluate(model, device, criterion, loader, epoch):

    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for data, labels in loader: # 生成一个 batch 的数据和标注
            data = data.to(device)
            labels = labels.to(device)
            outputs,_ = model(data) # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1) 
            preds = preds.cpu().numpy()
            print(preds)
            loss = criterion(outputs, labels) # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_test = {}
    log_test['epoch'] = epoch
    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro',zero_division=0)
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro',zero_division=0)
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro',zero_division=0)
    log_test['test_roc_auc_score'] = roc_auc_score(labels_list, preds_list, average='macro')

    return log_test

def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate
    r2 = r2_score(y_true, y_pred)

    # binary evaluate
    binary_acc = accuracy_score(binary_true, binary_pred)
    precision = precision_score(binary_true, binary_pred)
    recall = recall_score(binary_true, binary_pred)
    f1 = f1_score(binary_true, binary_pred)
    # y_pred evaluate
    roauc = roc_auc_score(binary_true, y_pred)
    prauc = average_precision_score(binary_true, y_pred)
    
    mcc = matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {
        'r2': r2,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roauc': roauc,
        'prauc':prauc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return result
