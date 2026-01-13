import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from IDRR_data import *
import numpy as np
import pandas as pd
from pathlib import Path as path

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import (Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer)
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict, Optional

# 设置可见的GPU设备
# 获取当前文件所在的目录和根目录
SRC_DIR = path(__file__).parent
ROOT_DIR = SRC_DIR.parent
OUTPUT_DIR = 'output_arg1'

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
            row['arg1'],
            add_special_tokens=True, 
            padding=True,
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

# === 保存评估结果函数 ===
def save_evaluation_results(
    trainer: Trainer,
    dataset: CustomDataset,
    label_list: list,
    output_filepath: path,
    dataset_name: str = 'eval',
    step: Optional[int] = None,
    ):
    """
    保存每条样本的评估结果为CSV格式
    
    Args:
        trainer: Trainer实例
        dataset: 评估数据集
        label_list: 标签列表
        output_filepath: 输出文件路径
        dataset_name: 数据集名称（用于区分dev/test）
        step: 训练步数（用于区分不同checkpoint的评估结果）
    """
    # 获取预测结果
    predictions = trainer.predict(dataset)
    pred_logits = predictions.predictions  # [n_samples, n_labels]
    
    # 获取真实标签
    true_labels = predictions.label_ids  # [n_samples, n_labels]
    
    # 处理预测结果
    pred_logits = pred_logits[..., :len(label_list)]
    true_labels = true_labels[..., :len(label_list)]
    
    # 计算预测标签（取最大概率的类别）
    pred_indices = np.argmax(pred_logits, axis=1)
    pred_probs = np.exp(pred_logits) / np.sum(np.exp(pred_logits), axis=1, keepdims=True)  # softmax概率
    
    # 获取真实标签索引
    true_label_indices = np.argmax(true_labels, axis=1)
    
    # 构建结果DataFrame
    results = []
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx].copy()
        
        # 添加预测信息
        pred_label_id = pred_indices[idx]
        pred_label = label_list[pred_label_id]
        pred_prob = float(pred_probs[idx, pred_label_id])
        
        true_label_id = true_label_indices[idx]
        true_label = label_list[true_label_id]
        
        # 构建结果字典
        result_dict = {
            'sample_idx': idx,
            'arg1': row.get('arg1', ''),
            'true_label_id': int(true_label_id),
            'true_label': true_label,
            'pred_label_id': int(pred_label_id),
            'pred_label': pred_label,
            'pred_prob': pred_prob,
            'is_correct': int(pred_label_id == true_label_id),
        }
        
        # 添加所有类别的概率
        for label_idx, label in enumerate(label_list):
            result_dict[f'prob_{label}'] = float(pred_probs[idx, label_idx])
        
        # 保留原始数据的其他列（如果有）
        # for col in dataset.df.columns:
        #     if col not in result_dict:
        #         result_dict[col] = row[col]
        
        results.append(result_dict)
    
    # 保存为CSV
    results_df = pd.DataFrame(results)
    
    # 确保输出目录存在
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果有step，在文件名中包含step信息
    if step is not None:
        output_filepath = output_filepath.parent / f"{output_filepath.stem}_step{step}{output_filepath.suffix}"
    
    results_df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f'> 已保存评估结果到: {output_filepath}')
    print(f'> 样本总数: {len(results_df)}, 正确数: {results_df["is_correct"].sum()}, 准确率: {results_df["is_correct"].mean():.4f}')

# === callback ===
class CustomCallback(TrainerCallback):
    def __init__(
        self, 
        log_filepath=None,
        eval_dataset: Optional[CustomDataset] = None,
        label_list: Optional[list] = None,
        save_eval_results: bool = True,
    ):
        super().__init__()
        if log_filepath:
            self.log_filepath = log_filepath
        else:
            self.log_filepath = ROOT_DIR / OUTPUT_DIR / 'log.jsonl'
        
        self.eval_dataset = eval_dataset
        self.label_list = label_list
        self.save_eval_results = save_eval_results
        self.output_dir = ROOT_DIR / OUTPUT_DIR / 'eval_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = None  # 将在on_train_begin中设置
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时保存trainer引用"""
        self.trainer = kwargs.get('trainer')
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.log_filepath, 'a', encoding='utf8')as f:
            f.write(str(kwargs['logs'])+'\n')

    def on_evaluate(self, args, state, control, metrics: Dict[str, float], **kwargs):
        """每次评估时保存详细结果"""
        if self.save_eval_results and self.eval_dataset is not None and self.label_list is not None and self.trainer is not None:
            try:
                current_step = state.global_step
                output_filepath = self.output_dir / f'dev_step{current_step}_results.csv'
                
                save_evaluation_results(
                    trainer=self.trainer,
                    dataset=self.eval_dataset,
                    label_list=self.label_list,
                    output_filepath=output_filepath,
                    dataset_name='dev',
                    step=current_step,
                )
            except Exception as e:
                print(f'> 保存评估结果时出错: {e}')
                import traceback
                traceback.print_exc()

def main():
    # === data ===
    dfs = IDRRDataFrames(
        data_name='pdtb2',
        data_level='top',
        data_relation='Implicit',
        data_path='/data/sunwh/idrr/data/raw/pdtb2.p1.csv',
    )
    label_list = dfs.label_list
    
    print(len(label_list))

    # === model ===
    model_name_or_path = "/data/sunwh/pretrained_models/roberta-base"
    # model_name_or_path = '/data/sunwh/model/flan-t5-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=len(label_list))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # === args ===
    training_args = TrainingArguments(
        output_dir=ROOT_DIR/OUTPUT_DIR,
        overwrite_output_dir=True,
        run_name='roberta_large_arg1',
        report_to='tensorboard',
        
        # strategies of evaluation, logging, save
        # eval_strategy = "epoch", 
        # eval_steps = 500,
        save_steps = 0.5,
        logging_strategy = 'steps',
        logging_steps = 10,
        save_strategy = 'epoch',
        # max_steps=2,
        
        # optimizer and lr_scheduler
        optim = 'adamw_torch',
        # optim = 'sgd',
        learning_rate = 2e-5,
        weight_decay = 0.01,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.05,
        
        # epochs and batches 
        num_train_epochs = 10, 
        # max_steps = args.max_steps,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 1,
        
        # train consumption
        eval_accumulation_steps=10,
        bf16=False,
        fp16=True,
    )

    # 加载训练集、验证集和测试集
    train_dataset = CustomDataset(dfs.train_df, label_list, tokenizer)
    dev_dataset = CustomDataset(dfs.dev_df, label_list, tokenizer)
    # test_dataset = CustomDataset(dfs.test_df, label_list, tokenizer)
    # print(test_dataset[0])
    # exit()

    # === train ===
    # 创建评估结果保存目录
    eval_results_dir = ROOT_DIR / OUTPUT_DIR / 'eval_results'
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=ComputeMetrics(dfs.label_list),
        callbacks=[CustomCallback(
            eval_dataset=dev_dataset,
            label_list=label_list,
            save_eval_results=True,
        )],
    )

    # 开始训练和评估
    train_result = trainer.train()
    print(f'\n> train_result:\n  {train_result}')
    
    # 训练结束后，保存最终验证集评估结果
    print('\n> 保存验证集评估结果...')
    save_evaluation_results(
        trainer=trainer,
        dataset=dev_dataset,
        label_list=label_list,
        output_filepath=eval_results_dir / 'dev_final_results.csv',
        dataset_name='dev',
    )
    
    # 评估测试集并保存结果
    # print('\n> 评估测试集并保存结果...')
    # test_result = trainer.evaluate(eval_dataset=dev_dataset)
    # print(f'\n> test_result:\n  {test_result}')
    # save_evaluation_results(
    #     trainer=trainer,
    #     dataset=test_dataset,
    #     label_list=label_list,
    #     output_filepath=eval_results_dir / 'test_final_results.csv',
    #     dataset_name='test',
    # )
    pass

if __name__ == '__main__':
    main()