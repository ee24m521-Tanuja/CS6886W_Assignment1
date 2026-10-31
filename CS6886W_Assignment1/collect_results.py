import glob,json,csv,os
rows=[]
for f in glob.glob('results/*/result.json'):
    with open(f) as h: r=json.load(h)
    r['run']=os.path.basename(os.path.dirname(f)); rows.append(r)
rows.sort(key=lambda x:x['best_val_accuracy'],reverse=True)
with open('results_summary.csv','w',newline='') as f:
    fields=['run','activation','optimizer','epochs','batch_size','learning_rate','seed','best_val_accuracy','best_epoch','final_val_accuracy']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print('\nTop runs:')
for r in rows[:10]: print(r['run'], 'best=',round(r['best_val_accuracy'],3),'epoch=',r['best_epoch'])
