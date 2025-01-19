import os
import statistics
import json
import math
import sys
import numpy as np
from pathlib import Path

creating_data_dir = 'creating_data'
poor_data_dir = 'poor_data'

def print_statics(path):
    data_num_info = []
    if Path(path).is_dir():
        for file in os.listdir(path):
            with open(path+'/'+file, 'r', encoding='utf-8') as f:
                data_num_info.append(len(json.load(f)['data']))
        print(path)
        num_recp = len(data_num_info)
        print('  num of recipients: {}'.format(num_recp))
        num_data = sum(data_num_info)
        print('  sum: {}'.format(num_data))
        print('  max: {}'.format(max(data_num_info)))
        print('  min: {}'.format(min(data_num_info)))
        ave = num_data / num_recp
        print('  ave: {}'.format(ave))
        print('  med: {}'.format(statistics.median(data_num_info)))
        pvar = statistics.pvariance(data_num_info, ave)
        print('  pvar: {}'.format(pvar))
        print('  pstd: {}'.format(math.sqrt(pvar)))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            num_data = len(json.load(f)['data'])
        print(path)
        print('  sum: {}'.format(num_data))

def label_statistics(dir):
    sum_label = np.zeros(8)
    argmax_sum_label = np.zeros(8)

    for file in os.listdir(dir):
        data = {}
        with open(dir+'/'+file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'data' in data:
            for talk in data['data']:
                label = talk['label']
                sum_label += np.array(label)
                argmax_sum_label[label.index(max(label))] += 1
    print('label sum:')
    print(sum_label)
    print(sum_label.sum())
    print('argmax label sum:')
    print(argmax_sum_label)
    print(argmax_sum_label.sum())

if __name__ == "__main__":
    print_statics(sys.argv[1])
    label_statistics(sys.argv[1])