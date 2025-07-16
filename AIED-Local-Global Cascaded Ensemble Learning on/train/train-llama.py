import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(output_dir):
    # 1. 检查设备是否有可用的GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 1. 从单个文件读取数据
    df = pd.read_csv(r'./training_data/output_3.csv')  # 假设文件名是output_2.csv

    # 2. 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 3. 获取文本和标签
    texts = df['content'].tolist()
    labels = df['kn_id'].tolist()

    # 4. 按 9:1 分割为训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=26, stratify=labels
    )

    # 2. 初始化Tokenizer

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '../unspervised_model/llama-3.2-1B')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token_id = 128004
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    encodings2 = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    # 3. 创建PyTorch数据集
    class TextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset(encodings, train_labels)
    val_dataset = TextDataset(encodings2, val_labels)

    # 将类别转换为 numpy 数组
    unique_classes = np.array(list(set(train_labels)))

    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 5. 定义自定义Trainer，加入类别权重
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # 使用类别权重计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            
            return (loss, outputs) if return_outputs else loss

    # 6. 定义评估指标函数
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        results = classification_report(labels, predictions, digits=4)
        return {
            'accuracy': accuracy,
            'results': results,
            'f1': f1
        }

    # 7. 初始化模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=155).to(device)
    model.config.pad_token_id = 128004
    for param in model.parameters(): param.data = param.data.contiguous()
    # 8. 训练参数设置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=5e-5,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=2,
        fp16=True
    )

    # 9. 使用CustomTrainer并移到GPU
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
    )

    # 10. 训练并保存最佳模型
    trainer.train()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train-bert.py <output_dir>")
        sys.exit(1)
    output_dir = sys.argv[1]
    main(output_dir)
