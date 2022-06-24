import pandas as pd
import numpy as np
import copy
from sklearn import metrics
import re

chexpert_categories = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Scoliosis', 
 'Pleural Effusion', 'Pleural Other', 
 'Pneumothorax', 'Fracture', 'Emphysema', 'Pneumonia', 
 'Edema', 'Atelectasis', 'Consolidation', 'Airspace Opacity', 
 'Lung Lesion', 'Hernia', 'Calcinosis', 'Support Devices', 
 'Airspace Disease', 'Hypoinflation', 'Other Finding']

anno = {"Atelectasis": 10,
        "Lung Lesion": 13,
        "Pleural Effusion": 3,
        "Pleural Other": 4,
        "Scoliosis": 2,
        "Support Devices": 16,
        "Calcinosis": 15,
        "Hernia": 14,
        "Cardiomegaly": 1,
        "Consolidation": 11,
        "Pneumothorax": 5,
        "Hypoinflation": 18,
        "Airspace Disease": 17,
        "Other Finding": 19,
        "Enlarged Cardiomediastinum": 0,
        "Airspace Opacity": 12,
        "Pneumonia": 8,
        "Edema": 9,
        "Emphysema": 7,
        "Fracture": 6}

label = {"NEGATIVE": 1, "POSITIVE": 2, "UNCERTAIN": 0}

def parse(gt_flag):
    #获得数据
    filename = 'val_res_labeled.csv'
    if gt_flag:
        filename = 'val_gts_labeled.csv'
    dataframe = pd.read_csv(filename)
    reg = re.compile('\[.[\w\s]*....[\w\s]*....[\w\s]*....[\w\s]*.\]')
    regg = re.compile('\'[\w\s]*\'')
    reggg = re.compile('[\w\s]*')
    list = []
    for i in range(len(dataframe["attributes"])):
        temp = dataframe["attributes"][i][1:-1]
        x = reg.findall(temp)
        lis = []
        for h in x:
            line = regg.findall(h)
            for c in line:
                lis.append(reggg.findall(c)[1])
        list.append(lis)


    for i in list:
        for h in range(len(i)):
            if(h % 4 == 1):
                i[h] = anno[i[h]]
                i[h+1] = label[i[h+1]]

    out_list = []

    for i in range(len(list)):
        addlist = 20*[3]
        for h in range(len(list[i])):
            if(h % 4 == 1):
                addlist[list[i][h]] = list[i][h+1]
        out_list.append(addlist)

    names = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Scoliosis', 'Pleural Effusion', 'Pleural Other', 'Pneumothorax', 'Fracture', 'Emphysema', 'Pneumonia', 'Edema',
             'Atelectasis', 'Consolidation', 'Airspace Opacity', 'Lung Lesion', 'Hernia', 'Calcinosis', 'Support Devices', 'Airspace Disease', 'Hypoinflation', 'Other Finding']


    out_csv = pd.DataFrame(data=out_list, columns=names)
    out_csv.to_csv('temp_'+filename, index=False)

def evaluate_label(tar, pred, ignore_nan=False, avg_method='weighted'):
    """
    Return precision, recall, f1, and prevalence for a single label.
    """
    
    if ignore_nan:
        idx = ~(np.isnan(tar) | np.isnan(pred))
        pred = pred[idx]
        tar = tar[idx]

    # print(tar, pred)
    # print(idx)
    
    results = {
        'precision': np.nan,
        'recall': np.nan,
        'f1': np.nan,
        # 'positives': int(tar.sum())
    }
    
    # if results['positives'] == 0:
    #     # return NaN if no positive labels
    #     return results
    
    results['precision'] = metrics.precision_score(tar, pred, average=avg_method, zero_division=0)
    results['recall'] = metrics.recall_score(tar, pred, average=avg_method, zero_division=0)
    # results['f1'] = 2*(results['precision']*results['recall'])/(results['precision']+results['recall'])
    results['f1'] = metrics.f1_score(tar, pred, average=avg_method, zero_division=0)

    
    return results    
    
def get_scores(target, prediction, categories, avg_method, ignore_nan=False):
    
    
    results = {}
    for i, c in enumerate(categories):
        results[c] = evaluate_label(target[:, i], prediction[:, i], ignore_nan=ignore_nan, avg_method=avg_method)
    
    # convert to dataframe
    df = pd.DataFrame.from_dict(results, orient='index')
    
    return df

def evaluate_labels(df_truth, df_label, method='mention', avg_method='weighted'):
    categories = list(df_truth.columns)
    
    # create the matrix of 0s and 1s
    preds = copy.copy(df_label.values)
    targets = copy.copy(df_truth.values)
    avg = 'binary'
    
    if method == 'mention':
        # any mention is a 1
#         print(preds)
#         print(targets)
#         preds[np.isin(preds, [-1, 0, 1])] = 1
#         targets[np.isin(targets, [-1, 0, 1])] = 1

#         # no mention is a 0
#         preds[np.isnan(preds)] = 0
#         targets[np.isnan(targets)] = 0
        
        # do not ignore NaN (which we have set to 0 anyway)
        ignore_nan=False
    elif method == 'negation':
        # successful prediction of negation
        idxNonZero = preds != 0
        idxZero = preds == 0
        preds[idxNonZero] = 0
        preds[idxZero] = 1
        
        idxNonZero = targets != 0
        idxZero = targets == 0
        targets[idxNonZero] = 0
        targets[idxZero] = 1
        
        # ignore NaN values
        ignore_nan=True
    elif method == 'uncertain':
        # any non-uncertain prediction is 0
        preds[preds!= -1] = 0
        targets[targets != -1] = 0
        
        # any uncertain prediction is 1
        preds[preds == -1] = 1
        targets[targets == -1] = 1
        
        # ignore NaN
        ignore_nan=True

    elif method == 'multiClass':
        # idx = ~(np.isnan(targets) | np.isnan(preds))
        # preds = preds[idx]
        # targets = targets[idx]
        preds[np.where(np.isnan(preds))] = 2
        targets[np.where(np.isnan(targets))] = 2
        avg = avg_method
        ignore_nan = True

    else:
        raise ValueError(f'Unrecognized method {method}')
        
    df = get_scores(targets, preds, categories, avg_method=avg, ignore_nan=ignore_nan)
    
    return df


def compute_overall_metrics(df):
    results = {}
    for i, m in enumerate(['precision', 'recall', 'f1']):
        col = df.values[:, i]
        results[m] = col.sum() / len(col)
    
    return results


def eval(val_path, gt_path):
    #加上主索引列
    # chexpert
    df_chexpert = pd.read_csv(val_path)
    n = len(df_chexpert)+1
    nlist = range(1,n)
    df_chexpert['idx'] = nlist
    df_chexpert.set_index('idx', inplace=True)
#     df_chexpert.rename(columns={'Airspace Opacity': 'Lung Opacity'}, inplace=True)
    df_chexpert = df_chexpert[chexpert_categories]

    # ground truth
    gs = pd.read_csv(gt_path)
    n = len(gs)+1
    nlist = range(1,n)
    gs['id'] = nlist
    gs.set_index('id', inplace=True)
#     gs.rename(columns={'Airspace Opacity': 'Lung Opacity'}, inplace=True)
    gs = gs[chexpert_categories]
    return df_chexpert, gs

class Runner():
    def run(self):
        print('ours:')
        parse(True)
        parse(False)
        df_chexpert, gs = eval('temp_val_res_labeled.csv', 'temp_val_gts_labeled.csv')
        df = evaluate_labels(gs, df_chexpert, method='multiClass', avg_method='macro')

        for c in df.columns:
            if 'float' in str(df.dtypes[c]):
                df[c] = np.round(df[c], 3)

        print(df)
        print('macro average:')
        results = compute_overall_metrics(df)
        print('ours', results)