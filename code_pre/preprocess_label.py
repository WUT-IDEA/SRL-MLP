# -*- coding:utf-8 -*-

import pandas as pd
from os import path


with open(path.join(path.dirname(__file__), '..', 'data','labels_change.txt'),'r') as f:
    with open(path.join(path.dirname(__file__), '..', 'data','labels_trans2num.txt'),'w') as w:
        # w.write("'id','gender','age','area'\n")
        for line in f.readlines():
            feature = line.split("|")
            if feature[1]=='m':
                feature[1]='1'
            elif feature[1]=='f':
                feature[1]='0'
            else:
                print('unknown gender')
                exit(1)

            if feature[2] == '-1979':
                feature[2]='0'
            elif feature[2] == '1980-1989':
                feature[2]='1'
            elif feature[2] =='1990+':
                feature[2]='2'
            else:
                print('unknown age')
                exit(1)
            feature[3]='999'
            sep = ","
            w.write(sep.join(feature))
            w.write('\n')

label_file = pd.read_table(path.join(path.dirname(__file__), '..', 'data','labels_trans2num.txt'),
                           header=None, sep=',', encoding='utf-8')
print(label_file)
print(len(label_file))
