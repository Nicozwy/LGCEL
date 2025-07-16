import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from transformers import XLNetTokenizer,AutoTokenizer,BloomForCausalLM

# 加载BERT tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(r"pretrain-model/Conan-embedding")
model = BertForMaskedLM.from_pretrained(r"pretrain-model/Conan-embedding")

# 加载CSV数据
csv_path = "training_data/unsupervised_train.csv"  # 替换为你的CSV文件路径
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} 文件不存在！")

# 读取CSV并提取`content`列
df = pd.read_csv(csv_path)
if "content" not in df.columns:
    raise ValueError(f"CSV文件中没有找到`content`列！")

# 转换为Hugging Face的Dataset格式
dataset = Dataset.from_pandas(df[["content"]])

# 对数据进行分词和编码
def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["content"])

# 数据预处理（数据整理器）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="autodl-tmp/atempmodel/Conan-embedding",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=500,
    fp16 = True
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

# 开始训练
trainer.train()

# 保存模型和分词器
model.save_pretrained("autodl-tmp/unspervised_model/Conan-embedding")
tokenizer.save_pretrained("autodl-tmp/unspervised_model/Conan-embedding")

print("无监督训练完成，模型已保存！")
