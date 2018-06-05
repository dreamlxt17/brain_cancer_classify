# coding=utf-8

import numpy as np
import pandas as pd
import csv
from models import RF, GB
from random import shuffle
from sklearn.preprocessing import StandardScaler


csv_dir = '/all/DATA_PROCEING/training_edema.csv'


reader = csv.reader(open(csv_dir))
content = []
for i, con in enumerate(reader):
    if i==0:
        continue
    content.append(con)


iter=50
acc1=0
acc2=0
indices = range(92)
for i in range(iter):
    datas = []
    labels = []
    shuffle(indices)
    for i in indices:
        row = content[i]
        if i == 0:
            continue
        labels.append(int(row[1]))
        datas.append(np.array(row[4:]).astype(float))

    label = np.array(labels)
    data = np.array(datas)
    data = StandardScaler.fit_transform(data)

    m = 70
    tr_data, te_data = data[:m], data[m:]
    tr_label, te_label = label[:m], label[m:]
    print tr_data.shape

    clf = GB()
    clf.fit(tr_data, tr_label)
    tr_pred = clf.predict(tr_data)
    te_pred = clf.predict(te_data)

    tr_acc = np.sum(tr_pred==tr_label)*1.0/len(tr_label)
    te_acc = np.sum(te_pred==te_label)*1.0/len(te_label)

    acc1+=tr_acc
    acc2+=te_acc
print acc1/iter, acc2/iter