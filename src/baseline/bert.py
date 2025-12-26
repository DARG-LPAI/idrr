import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
from IDRR_data import *

import numpy as np
import pandas as pd
from pathlib import Path as path
from typing import Dict

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import (Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer)
from transformers import TrainerCallback, TrainerState, TrainerControl

# 设置可见的GPU设备
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
            padding=False,  # 不在单个样本中 padding，由 DataCollatorWithPadding 处理
            truncation='longest_first', 
            max_length=512,
        )
        model_inputs['labels'] = self.ys[index]
        return model_inputs
    
    def __len__(self):
        return self.df.shape[0]

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
        # pred: np.array [datasize, ]
        pred: np.array [datasize, n]
        labels: np.array [datasize, n]
        X[p][q]=True, sample p belongs to label q (False otherwise)
        """
        pred, labels = eval_pred
        pred: np.ndarray
        labels: np.ndarray
        
        pred = pred[..., :len(self.label_list)]
        labels = labels[..., :len(self.label_list)]
        
        # pred = pred!=0
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

# === callback ===
class CustomCallback(TrainerCallback):
    """统一处理训练日志记录与基于 F1 的最优模型追踪"""
    def __init__(
        self, 
        log_filepath=None,
    ):
        super().__init__()
        self.log_filepath = log_filepath or ROOT_DIR / 'output_dir' / 'log.jsonl'
        # 记录最优 F1 及对应 checkpoint 信息
        self.best_f1 = -1.0
        self.best_checkpoint = None
        self.best_step = None
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """追加保存 Trainer 的日志事件"""
        with open(self.log_filepath, 'a', encoding='utf8') as f:
            f.write(str(kwargs['logs']) + '\n')
    
    def on_evaluate(self, args, state, control, metrics: Dict[str, float], **kwargs):
        """在每次验证后基于 Macro-F1 追踪最优 checkpoint"""
        current_f1 = metrics.get('eval_Macro-F1', -1.0)
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_step = state.global_step
            self.best_checkpoint = path(args.output_dir) / f'checkpoint-{state.global_step}'
            print(f'> 发现更好的模型: Step {self.best_step}, F1={current_f1:.4f}')
            print(f'> 最佳checkpoint: {self.best_checkpoint}')


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
    # model_name_or_path = '/data/sunwh/model/flan-t5-base'
    model_name_or_path = '../pretrained_models/roberta-base'
    model_name_or_path = '../pretrained_models/Qwen/Qwen3-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # 如果 eos_token 也不存在，添加一个新的 pad_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 确保模型的 config 中也设置了 pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path , 
        num_labels=len(label_list),
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 同步更新模型 config（双重保险）
    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f'> pad_token: {tokenizer.pad_token}')
    print(f'> pad_token_id: {tokenizer.pad_token_id}')

    # === args ===
    training_args = TrainingArguments(
        output_dir=ROOT_DIR/'output_dir',
        overwrite_output_dir=True,
        run_name='sft_cls',
        report_to='swanlab',
        
        # strategies of evaluation, logging, save
        eval_strategy = "epoch", 
        eval_steps = 100,
        logging_strategy = 'steps',
        logging_steps = 10,
        save_strategy = 'epoch',
        save_steps = 1,
        # max_steps=2,
        
        # optimizer and lr_scheduler
        optim = 'adamw_torch',
        # optim = 'sgd',
        learning_rate = 1e-5,
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.05,
        
        # epochs and batches 
        num_train_epochs = 10, 
        # max_steps = args.max_steps,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 1,
        
        # train consumption
        eval_accumulation_steps=10,
        bf16=True,
        fp16=False,

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
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=ComputeMetrics(dfs.label_list),
        callbacks=[(best_log_callback := CustomCallback())],
    )

    # 开始训练和评估
    print('\n> 开始训练模型...')
    train_result = trainer.train()
    print(f'\n> 训练完成:\n  {train_result}')

    if best_log_callback.best_checkpoint:
        print(f'\n> 最佳模型checkpoint: {best_log_callback.best_checkpoint}')
        print(f'> 最佳F1分数: {best_log_callback.best_f1:.4f}')
    else:
        print('\n> 使用训练结束时的模型进行评估')

    print('\n> 开始评估模型...')
    test_result = trainer.evaluate(eval_dataset=test_dataset)
    print(f'> test_result:\n  {test_result}')

    pass

if __name__ == '__main__':
    main()