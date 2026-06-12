import argparse
import os
import json
import numpy as np
import fiona
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def read_labels(shp_path, label_field='label', pred_field='pred'):
    y_true=[]
    y_pred=[]
    with fiona.open(shp_path,'r') as src:
        for feat in src:
            props=feat['properties']
            if label_field in props and pred_field in props:
                y_true.append(int(props[label_field]))
                y_pred.append(int(props[pred_field]))
    return np.array(y_true), np.array(y_pred)

def save_confusion_matrix(cm, out_csv):
    os.makedirs(os.path.dirname(out_csv),exist_ok=True)
    with open(out_csv,'w',encoding='utf-8') as f:
        for row in cm:
            f.write(','.join(map(str,row))+'\n')

def save_class_report(report_dict, class_names, out_csv):
    os.makedirs(os.path.dirname(out_csv),exist_ok=True)
    keys=[str(i) for i in range(len(class_names))]
    with open(out_csv,'w',encoding='utf-8') as f:
        f.write('class,precision,recall,f1-score,support,name\n')
        for k in keys:
            if k in report_dict:
                r=report_dict[k]
                f.write(f"{k},{r['precision']:.6f},{r['recall']:.6f},{r['f1-score']:.6f},{r['support']},{class_names[int(k)]}\n")
        # overall
        acc=report_dict.get('accuracy',0)
        f.write(f"overall_accuracy,{acc:.6f}\\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predicted_shp',type=str,required=True)
    p.add_argument('--label_json',type=str,default=None)
    p.add_argument('--label_field',type=str,default='label')
    p.add_argument('--pred_field',type=str,default='pred')
    p.add_argument('--out_confusion',type=str,default='outputs/confusion_matrix.csv')
    p.add_argument('--out_report',type=str,default='outputs/class_report.csv')
    a=p.parse_args()
    names=None
    if a.label_json and os.path.exists(a.label_json):
        with open(a.label_json,'r',encoding='utf-8') as f:
            info=json.load(f)
            names=info.get('label_names',None)
    y_true,y_pred=read_labels(a.predicted_shp,label_field=a.label_field,pred_field=a.pred_field)
    if y_true.size==0:
        raise RuntimeError('No labels found in shapefile')
    C=max(y_true.max(),y_pred.max())+1
    cm=confusion_matrix(y_true,y_pred,labels=list(range(C)))
    save_confusion_matrix(cm,a.out_confusion)
    acc=accuracy_score(y_true,y_pred)
    report=classification_report(y_true,y_pred,output_dict=True,labels=list(range(C)))
    if names is None:
        names=[str(i) for i in range(C)]
    save_class_report(report,names,a.out_report)
    print(f"overall_accuracy={acc:.6f}")

if __name__=='__main__':
    main()

