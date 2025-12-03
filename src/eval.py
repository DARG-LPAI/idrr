import random
import re
import string
import argparse

from pathlib import Path as path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, classification_report

from utils.mylogger import logger
from utils.utils import read_file, write_file

LABEL_LIST = [
    'Comparison',
    'Contingency',
    'Expansion',
    'Temporal',
]
SEC_LABEL_LIST = [
    'Concession',
    'Contrast',
    'Cause',
    'Pragmatic cause',
    'Alternative',
    'Conjunction',
    'List',
    'Instantiation',
    'Restatement',
    'Asynchronous',
    'Synchrony'
]

def extract_label(pred):
    if '</think>' in pred:
        pred = pred.split('</think>')[-1]
    elif 'Relation: ' in pred:
        pred = pred.split('Relation: ')[-1].split('.')[0]
    for label in LABEL_LIST:
        if label in pred:
            return label
    raise ValueError(f"Label {pred} not found in {LABEL_LIST}, pred: \n-----\n{pred}\n-------\n")

def extract_answer(s):
    # matches = re.findall(r'<answer[^>]*>(.*?)<\/answer>', s, re.DOTALL)
    # return matches[0] if matches else "NONE!"
    return s.split('->')[-1]

def find_most_frequent_word(text):
    tags = ['Contingency', 'Expansion', 'Temporal', 'Comparison']
    words = re.findall(r"[\w']+", text)
    cleaned_words = [word.strip(string.punctuation) for word in words]

    
    counts = defaultdict(int)
    for word in cleaned_words:
        if word in tags:
            counts[word] += 1
    
    if not counts:
        # print('*'*20)
        # print(cleaned_words)
        # print('*'*20)
        return random.choice(tags)
        return "None!"
    
    max_count = max(counts.values())
    max_tags = [tag for tag, cnt in counts.items() if cnt == max_count]
    
    return max_tags[0]
    if len(max_tags) > 1:
        return tuple(sorted(max_tags))
    else:
        return max_tags[0]

def eval(gen_preds, data_path):
    # data_path是具体文件的话，提取目录
    if '.json' in data_path:
        data_path = '/'.join(data_path.split('/')[:-1])

    labels, preds = [], []
    cnt = 0
    for it in gen_preds:
        # it['label'] = extract_answer(it['label'])
        # pred = extract_answer(it['predict'])
        pred = extract_label(it['predict'])
        label = extract_label(it['label'])
        it['pred_label'] = pred
        preds.append(pred)
        labels.append(label)
        cnt += 1

    write_file(path(data_path) / 'predictions.json', gen_preds)
    precision, recall, f1_score, support = precision_recall_fscore_support(labels, preds, average='macro')
    acc = sum([1 for i in range(len(labels)) if labels[i] == preds[i]]) / len(labels)
    acc_test = [pred == label for pred, label in zip(preds, labels)].count(True) / len(labels)
    assert acc == acc_test
    write_file(path(data_path) / 'eval_results.json', {'F1': f1_score,  'Accuracy': acc})
    # 打印结果
    logger.info("\n***分类报告 (Classification Report)***")
    print(f"F1分数 (F1 Score):{f1_score}")
    print(f"准确率 (Accuracy):{acc}")
    print(f"精确度 (Precision):{precision}")
    print(f"召回率 (Recall):{recall}")
    logger.info("\n***详细分类报告 (Detailed Classification Report)***")
    print(classification_report(labels, preds, target_names=LABEL_LIST))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    data_path = args.data_path
    if '.json' in data_path:
        gen_preds = read_file(data_path)
    else:
        gen_preds = read_file(path(data_path) / 'generated_predictions.jsonl')

    eval(gen_preds, data_path)

if __name__ == '__main__':
    main()