import os
# 设置可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
from IDRR_data import *

import numpy as np
import pandas as pd
from pathlib import Path as path
import json

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import (Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer)
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict, Optional

# 获取当前文件所在的目录和根目录
SRC_DIR = path(__file__).parent
ROOT_DIR = SRC_DIR.parent

# === dataset ===
class CustomDataset(Dataset):
    def __init__(self, df, label_list, tokenizer) -> None:
        self.df:pd.DataFrame = df
        label_num = len(label_list)
        self.id2label = {i:label for i, label in enumerate(label_list)}
        self.ys = np.eye(label_num, label_num)[self.df['label11id']]
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        model_inputs = self.tokenizer(
            row['arg1'], row['arg2'],
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=512,
        )
        model_inputs['labels'] = self.ys[index]
        return model_inputs
    
    def __len__(self):
        return self.df.shape[0]

# === 扩展测试集Dataset ===
class ExtendedTestDataset(Dataset):
    """
    扩展测试集：每个原始样本生成三种变体
    1. arg1 + arg2 (原始)
    2. arg1 + "" (只包含arg1)
    3. "" + arg2 (只包含arg2)
    """
    def __init__(self, df, label_list, tokenizer) -> None:
        self.df:pd.DataFrame = df
        label_num = len(label_list)
        self.id2label = {i:label for i, label in enumerate(label_list)}
        self.tokenizer = tokenizer
        
        # 为每个原始样本生成三种变体
        self.extended_samples = []
        self.original_indices = []  # 保存每个扩展样本对应的原始DataFrame索引
        self.variant_types = []  # 保存每个扩展样本的变体类型
        
        for idx, row in df.iterrows():
            original_label = row['label11id']
            # 1. 原始: arg1 + arg2
            self.extended_samples.append({
                'arg1': row['arg1'],
                'arg2': row['arg2'],
                'label11id': original_label,
            })
            self.original_indices.append(idx)
            self.variant_types.append('both')
            
            # 2. 只arg1: arg1 + ""
            self.extended_samples.append({
                'arg1': row['arg1'],
                'arg2': '',
                'label11id': original_label,
            })
            self.original_indices.append(idx)
            self.variant_types.append('arg1_only')
            
            # 3. 只arg2: "" + arg2
            self.extended_samples.append({
                'arg1': '',
                'arg2': row['arg2'],
                'label11id': original_label,
            })
            self.original_indices.append(idx)
            self.variant_types.append('arg2_only')
        
        # 构建标签矩阵
        self.ys = np.eye(label_num, label_num)[[s['label11id'] for s in self.extended_samples]]
    
    def __getitem__(self, index):
        sample = self.extended_samples[index]
        model_inputs = self.tokenizer(
            sample['arg1'], sample['arg2'],
            add_special_tokens=True, 
            padding=True,
            truncation='longest_first', 
            max_length=512,
        )
        model_inputs['labels'] = self.ys[index]
        return model_inputs
    
    def __len__(self):
        return len(self.extended_samples)

# === metric ===
class ComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.metric_names = ['Macro-F1', 'Acc']
    
    def __call__(self, eval_pred):
        """
        n = label categories
        eval_pred: (pred, labels)
        pred: np.array [datasize, n]
        labels: np.array [datasize, n]
        """
        pred, labels = eval_pred
        pred: np.ndarray
        labels: np.ndarray
        
        pred = pred[..., :len(self.label_list)]
        labels = labels[..., :len(self.label_list)]
        
        max_indices = np.argmax(pred, axis=1)
        bpred = np.zeros_like(pred, dtype=int)
        bpred[np.arange(pred.shape[0]), max_indices] = 1
        pred = bpred
        assert ( pred.sum(axis=1)<=1 ).sum() == pred.shape[0]
        labels = labels!=0
        
        res = {
            'Macro-F1': f1_score(labels, pred, average='macro', zero_division=0),
            'Acc': np.sum(pred*labels)/len(pred),
        } 
        return res

# === callback for best model selection ===
class BestModelCallback(TrainerCallback):
    """基于F1选择最佳模型的回调"""
    def __init__(self):
        super().__init__()
        self.best_f1 = -1.0
        self.best_checkpoint = None
        self.best_step = None
    
    def on_evaluate(self, args, state, control, metrics: Dict[str, float], **kwargs):
        """每次评估时检查是否为最佳模型"""
        current_f1 = metrics.get('eval_Macro-F1', -1.0)
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_step = state.global_step
            # 最佳checkpoint路径
            self.best_checkpoint = path(args.output_dir) / f'checkpoint-{state.global_step}'
            print(f'> 发现更好的模型: Step {self.best_step}, F1={current_f1:.4f}')
            print(f'> 最佳checkpoint: {self.best_checkpoint}')

# === callback ===
class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        log_filepath=None,
    ):
        super().__init__()
        if log_filepath:
            self.log_filepath = log_filepath
        else:
            self.log_filepath = ROOT_DIR / 'output_dir' / 'log.jsonl'
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.log_filepath, 'a', encoding='utf8')as f:
            f.write(str(kwargs['logs'])+'\n')

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        pass

def evaluate_extended_testset(trainer, extended_dataset, label_list, original_test_df):
    """
    在扩展测试集上评估，并分类结果
    
    Args:
        trainer: Trainer实例
        extended_dataset: ExtendedTestDataset实例
        label_list: 标签列表
        original_test_df: 原始测试集DataFrame
    """
    print('\n> 开始在扩展测试集上评估...')
    
    # 获取预测结果
    predictions = trainer.predict(extended_dataset)
    pred_logits = predictions.predictions  # [n_samples, n_labels]
    true_labels = predictions.label_ids  # [n_samples, n_labels]
    
    # 处理预测结果
    pred_logits = pred_logits[..., :len(label_list)]
    true_labels = true_labels[..., :len(label_list)]
    
    # 计算预测标签
    pred_indices = np.argmax(pred_logits, axis=1)
    true_label_indices = np.argmax(true_labels, axis=1)
    
    # 组织结果：按原始样本分组
    n_original = len(original_test_df)
    results_by_sample = {}
    
    for idx in range(len(extended_dataset)):
        # 从ExtendedTestDataset获取原始索引和变体类型
        original_idx = extended_dataset.original_indices[idx]
        variant_type = extended_dataset.variant_types[idx]
        
        if original_idx not in results_by_sample:
            results_by_sample[original_idx] = {
                'both': None,
                'arg1_only': None,
                'arg2_only': None,
            }
        
        is_correct = int(pred_indices[idx] == true_label_indices[idx])
        pred_label_id = int(pred_indices[idx])
        true_label_id = int(true_label_indices[idx])
        
        results_by_sample[original_idx][variant_type] = {
            'is_correct': is_correct,
            'pred_label_id': pred_label_id,
            'true_label_id': true_label_id,
        }
    
    # 确定正确子集：arg1+arg2预测正确的样本
    correct_subset = []
    for original_idx, results in results_by_sample.items():
        if results['both'] and results['both']['is_correct']:
            correct_subset.append(original_idx)
    
    print(f'> 正确子集大小: {len(correct_subset)} / {n_original}')
    
    # 分类：evidence_one_sided 和 evidence_cross_argument
    evidence_one_sided = []
    evidence_cross_argument = []
    
    for original_idx in correct_subset:
        results = results_by_sample[original_idx]
        arg1_correct = results['arg1_only']['is_correct'] if results['arg1_only'] else False
        arg2_correct = results['arg2_only']['is_correct'] if results['arg2_only'] else False
        
        # 只凭arg1或arg2就能做对
        if arg1_correct or arg2_correct:
            evidence_one_sided.append(original_idx)
        # 只凭arg1或arg2都做不对
        else:
            evidence_cross_argument.append(original_idx)
    
    print(f'\n> 分类结果统计:')
    print(f'  - evidence_one_sided: {len(evidence_one_sided)} 个样本')
    print(f'  - evidence_cross_argument: {len(evidence_cross_argument)} 个样本')
    
    # 统计各类别分布
    def print_label_distribution(indices, name):
        if len(indices) == 0:
            print(f'\n  {name} 标签分布: 无样本')
            return
        
        labels = original_test_df.loc[indices, 'label11id'].values
        label_counts = pd.Series(labels).value_counts().sort_index()
        
        print(f'\n  {name} 标签分布 (总数: {len(indices)}):')
        for label_id, count in label_counts.items():
            label_name = label_list[label_id] if label_id < len(label_list) else f'Unknown({label_id})'
            print(f'    {label_name}: {count} 个 ({count/len(indices)*100:.2f}%)')
    
    print_label_distribution(evidence_one_sided, 'evidence_one_sided')
    print_label_distribution(evidence_cross_argument, 'evidence_cross_argument')
    
    # 保存结果
    output_dir = ROOT_DIR / 'output_dir' / 'extended_eval_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存evidence_one_sided
    one_sided_df = original_test_df.loc[evidence_one_sided].copy()
    one_sided_df.reset_index(drop=True, inplace=True)
    one_sided_path = output_dir / 'evidence_one_sided.csv'
    one_sided_df.to_csv(one_sided_path, index=False, encoding='utf-8')
    print(f'\n> 已保存 evidence_one_sided 到: {one_sided_path}')
    
    # 保存evidence_cross_argument
    cross_arg_df = original_test_df.loc[evidence_cross_argument].copy()
    cross_arg_df.reset_index(drop=True, inplace=True)
    cross_arg_path = output_dir / 'evidence_cross_argument.csv'
    cross_arg_df.to_csv(cross_arg_path, index=False, encoding='utf-8')
    print(f'> 已保存 evidence_cross_argument 到: {cross_arg_path}')
    
    # 保存详细统计信息
    stats = {
        'total_samples': n_original,
        'correct_subset_size': len(correct_subset),
        'evidence_one_sided_size': len(evidence_one_sided),
        'evidence_cross_argument_size': len(evidence_cross_argument),
        'evidence_one_sided_ratio': len(evidence_one_sided) / len(correct_subset) if len(correct_subset) > 0 else 0,
        'evidence_cross_argument_ratio': len(evidence_cross_argument) / len(correct_subset) if len(correct_subset) > 0 else 0,
    }
    
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'> 已保存统计信息到: {stats_path}')
    
    return evidence_one_sided, evidence_cross_argument

def main():
    # === data ===
    dfs = IDRRDataFrames(
        data_name='pdtb2',
        data_level='top',
        data_relation='Implicit',
        data_path='/data/sunwh/idrr/data/raw/pdtb2.p1.csv',
    )
    label_list = dfs.label_list
    
    print(f'> 标签数量: {len(label_list)}')
    print(f'> 标签列表: {label_list}')

    # === model ===
    model_name_or_path = '/data/sunwh/idrr/src/output_dir/checkpoint-3555'
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, 
        num_labels=len(label_list)
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # === args ===
    training_args = TrainingArguments(
        output_dir=ROOT_DIR/'output_dir',
        overwrite_output_dir=True,
        run_name='roberta_large_extended_eval',
        
        # strategies of evaluation, logging, save
        eval_strategy = "epoch",  # 改为epoch策略，便于选择最佳模型
        logging_strategy = 'steps',
        logging_steps = 10,
        save_strategy = 'epoch',  # 每个epoch保存一次
        save_total_limit = 10,  # 最多保存10个checkpoint
        
        # optimizer and lr_scheduler
        optim = 'adamw_torch',
        learning_rate = 2e-5,
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.05,
        
        # epochs and batches 
        num_train_epochs = 10, 
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 1,
        
        # train consumption
        eval_accumulation_steps=10,
        bf16=False,
        fp16=True,
        
        # 加载最佳模型
        load_best_model_at_end=True,
        metric_for_best_model='eval_Macro-F1',
        greater_is_better=True,
    )

    # 加载训练集、验证集和测试集
    train_dataset = CustomDataset(dfs.train_df, label_list, tokenizer)
    dev_dataset = CustomDataset(dfs.dev_df, label_list, tokenizer)
    test_dataset = CustomDataset(dfs.test_df, label_list, tokenizer)

    # === train ===
    best_model_callback = BestModelCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=ComputeMetrics(dfs.label_list),
        callbacks=[CustomCallback(), best_model_callback],
    )

    # 开始训练
    print('\n> 开始训练模型...')
    # train_result = trainer.train()
    # print(f'\n> 训练完成:\n  {train_result}')
    
    # 训练结束后，trainer会自动加载最佳模型（因为设置了load_best_model_at_end=True）
    if best_model_callback.best_checkpoint:
        print(f'\n> 最佳模型checkpoint: {best_model_callback.best_checkpoint}')
        print(f'> 最佳F1分数: {best_model_callback.best_f1:.4f}')
    else:
        print('\n> 使用训练结束时的模型进行评估')
    
    # 创建扩展测试集
    print('\n> 创建扩展测试集...')
    extended_test_dataset = ExtendedTestDataset(dfs.test_df, label_list, tokenizer)
    print(f'> 扩展测试集大小: {len(extended_test_dataset)} (原始测试集: {len(dfs.test_df)})')
    
    # 在扩展测试集上评估并分类
    evidence_one_sided, evidence_cross_argument = evaluate_extended_testset(
        trainer=trainer,
        extended_dataset=extended_test_dataset,
        label_list=label_list,
        original_test_df=dfs.test_df,
    )
    
    print('\n> 所有任务完成！')

if __name__ == '__main__':
    main()

