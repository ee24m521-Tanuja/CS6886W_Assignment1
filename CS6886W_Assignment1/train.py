import argparse,csv,json,os,random
import numpy as np
import torch
import torch.nn as nn
from data import get_loaders
from model import VGG6
from optimizers import create_optimizer
from engine import run_epoch

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--activation',choices=['relu','sigmoid','tanh','silu','gelu'],default='relu')
    p.add_argument('--optimizer',choices=['sgd','nesterov','adam','adagrad','rmsprop','nadam'],default='adam')
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--output',default='results/run')
    p.add_argument('--wandb',action='store_true'); p.add_argument('--project',default='cs6886w-a1-vgg6')
    a=p.parse_args(); seed_all(a.seed); os.makedirs(a.output,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    tr,va=get_loaders(a.batch_size,workers=a.workers); model=VGG6(a.activation).to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=create_optimizer(a.optimizer,model.parameters(),a.lr)
    wb=None
    if a.wandb:
        import wandb; wb=wandb.init(project=a.project,config=vars(a))
    rows=[]; best=-1.; best_epoch=0
    for e in range(1,a.epochs+1):
        tl,ta=run_epoch(model,tr,criterion,device,optimizer); vl,va_acc=run_epoch(model,va,criterion,device)
        row={'epoch':e,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':va_acc}; rows.append(row)
        if wb: wb.log(row,step=e)
        if va_acc>best:
            best,best_epoch=va_acc,e
            torch.save({'model_state_dict':model.state_dict(),'config':vars(a),'best_val_accuracy':best,'best_epoch':e},os.path.join(a.output,'best_model.pth'))
        print(f'Epoch {e:03d}/{a.epochs} train_loss={tl:.4f} train_acc={ta:.2f}% val_loss={vl:.4f} val_acc={va_acc:.2f}%')
    with open(os.path.join(a.output,'history.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    result={'activation':a.activation,'optimizer':a.optimizer,'epochs':a.epochs,'batch_size':a.batch_size,'learning_rate':a.lr,'seed':a.seed,'best_val_accuracy':best,'best_epoch':best_epoch,'final_val_accuracy':rows[-1]['val_accuracy']}
    with open(os.path.join(a.output,'result.json'),'w') as f: json.dump(result,f,indent=2)
    if wb: wb.summary.update(result); wb.finish()
if __name__=='__main__': main()
