# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from transformers import EarlyStoppingCallback
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.utils.class_weight import compute_class_weight
# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # 1. 从单个文件读取数据
# df = pd.read_csv(r'./training_data/output_3.csv')  # 假设文件名是output_3.csv

# # 2. 打乱数据
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # 3. 获取文本和标签
# texts = df['content'].tolist()
# labels = df['kn_id'].tolist()

# # 4. 按 9:1 分割为训练集和验证集
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     texts, labels, test_size=0.1, random_state=20, stratify=labels
# )

# # 5. 数据预处理
# tokenizer = AutoTokenizer.from_pretrained(r"pretrain-model/deepseek")

# # 设置 padding token，如果没有默认的 padding token
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 padding token

# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# # 6. 创建PyTorch数据集
# class TextDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = TextDataset(train_encodings, train_labels)
# val_dataset = TextDataset(val_encodings, val_labels)

# # 7. 计算类别权重
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train_labels = np.array(train_labels)  # 确保 train_labels 是 NumPy 数组
# unique_classes = np.array(sorted(set(train_labels)))  # 确保类别有序
# class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# # 8. 定义自定义Trainer，加入类别权重
# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels")
#         # Forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # 使用类别权重计算交叉熵损失
#         loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
#         loss = loss_fct(logits, labels)
        
#         return (loss, outputs) if return_outputs else loss

# # 9. 定义评估指标函数
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(axis=-1)
#     accuracy = accuracy_score(labels, predictions)
#     f1 = f1_score(labels, predictions, average='macro')
#     return {
#         'accuracy': accuracy,
#         'f1': f1
#     }

# # 10. 加载模型
# model = AutoModelForSequenceClassification.from_pretrained(r"pretrain-model/deepseek", num_labels=155).to(device)

# # 11. 训练模型
# training_args = TrainingArguments(
#     output_dir=r'autodl-tmp/deepseek',
#     num_train_epochs=35,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=40,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
#     gradient_accumulation_steps=1024,
#     fp16=True,
#     optim="adamw_bnb_8bit"  # 使用 badam 优化器
# )

# # 12. 使用CustomTrainer
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
# )

# # 13. 训练并保存最佳模型
# trainer.train()

# # 14. 评估并生成classification_report
# eval_results = trainer.predict(val_dataset)

# # 获取预测的标签和真实的标签
# y_pred = eval_results.predictions.argmax(axis=-1)
# y_true = eval_results.label_ids

# # 获取训练和验证集中所有唯一的标签
# all_labels = sorted(set(train_labels.tolist() + val_labels))

# # 生成分类报告
# report = classification_report(y_true, y_pred, target_names=[str(k) for k in all_labels])

# # 打印分类报告
# print(report)

# # 保存报告到文件
# with open('autodl-tmp/class/classification_report_deepseek.txt', 'w') as f:
#     f.write(report)


###lora微调
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 读取数据
df = pd.read_csv(r'./training_data/output_3.csv')  # 假设文件名是 output_3.csv

# 2. 打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. 获取文本和标签
texts = df['content'].tolist()
labels = df['kn_id'].tolist()

# 4. 按 9:1 分割为训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=20, stratify=labels
)

# 5. 数据预处理
tokenizer = AutoTokenizer.from_pretrained(r"pretrain-model/deepseek-math-7b-rl")

# 设置 padding token，如果没有默认的 padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 padding token

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# 6. 创建 PyTorch 数据集
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

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# 7. 计算类别权重
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

unique_classes = np.array(sorted(set(train_labels)))  # 确保类别有序
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 8. 加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(
    r"pretrain-model/deepseek-math-7b-rl", num_labels=155
).to(device)
model.config.pad_token_id = model.config.eos_token_id
# 9. 配置 LoRA 适配层
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # 任务类型：文本分类
    r=8,  # LoRA 低秩维度
    lora_alpha=32,  # LoRA 缩放因子
    lora_dropout=0.05,  # LoRA Dropout
    bias="none"
)

# 10. 将模型转换为 LoRA 版本
model = get_peft_model(model, lora_config)

# 11. 训练参数设置
training_args = TrainingArguments(
    output_dir=r'autodl-tmp/deepseek_7B_LoRA',
    num_train_epochs=35,  # 适当减少 LoRA 训练轮数
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=40,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    gradient_accumulation_steps=8,  # 适配 batch size
    fp16=True,  # 开启混合精度训练
    optim="adamw_torch"  # 使用适配 LoRA 的优化器
)

# 12. 定义自定义 Trainer（加权损失）
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 13. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': accuracy, 'f1': f1}

# 14. 训练模型
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
)

trainer.train()

# 15. 评估模型
eval_results = trainer.predict(val_dataset)

# 16. 生成分类报告
y_pred = eval_results.predictions.argmax(axis=-1)
y_true = eval_results.label_ids
all_labels = sorted(set(train_labels + val_labels))
report = classification_report(y_true, y_pred, target_names=[str(k) for k in all_labels])

# 17. 输出分类报告
print(report)

# 18. 保存报告
with open('autodl-tmp/class/classification_report_deepseek_7B_LoRA.txt', 'w') as f:
    f.write(report)