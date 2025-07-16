# coding=utf-8
import argparse
import re
from bs4 import BeautifulSoup
import csv
import sys
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from transformers import BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import LlamaForSequenceClassification, LlamaTokenizer, PreTrainedTokenizerFast
from transformers import Qwen2ForSequenceClassification, Qwen2TokenizerFast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
MAX_INT = sys.maxsize

import os
import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import softmax

os.environ["NCCL_SOCKET_IFNAME"] = "eth0"


# 预处理文本的函数
def preprocess_text(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()

    clean_text = clean_text.replace('$$', '')
    clean_text = re.sub(r'\s+', ' ', clean_text)
    # 替换 `\angle` 为 `∠`
    clean_text = re.sub(r'\\angle', '∠',clean_text)
    clean_text = re.sub(r'\\pi', 'π', clean_text)
    clean_text = re.sub(r'\\ldots', '...', clean_text)
    clean_text = re.sub(r'\\because', '∵', clean_text)
    clean_text = re.sub(r'\\therefore', '∴', clean_text)
    clean_text = re.sub(r'\\odot', '⊙', clean_text)
    clean_text = re.sub(r'\\bot', '⊥',clean_text)
    clean_text = re.sub(r'\\overset\frown', '弧', clean_text)
    clean_text = re.sub(r'\\cong', '≌', clean_text)
    clean_text = clean_text.replace('\;', '')
    clean_text = clean_text.replace('~', '')
    clean_text = clean_text.replace('\left[', '')
    clean_text = clean_text.replace('right]', '')
    clean_text = clean_text.replace('\/', '/')
    clean_text = clean_text.replace('\\tan', 'tan')
    clean_text = clean_text.replace('\\text', '')
    clean_text = clean_text.replace('\\triangle', '△')
    clean_text = clean_text.replace('\pm', '±')

    # 替换 `^\circ` 为 `°`
    clean_text = re.sub(r'\\circ', '°', clean_text)
    # 删除 `}^` 和 `{^`
    clean_text = re.sub(r'\^°', '°', clean_text)
    clean_text = re.sub(r'\^°', '°', clean_text)
    clean_text = re.sub(r'}°', '°', clean_text)
    clean_text = re.sub(r'{°', '°', clean_text)

    # 替换 \dfrac{分子}{分母} 为 分子/分母
    clean_text = re.sub(r'\\dfrac\{(.+?)\}\{(.+?)\}', r'\1/\2',clean_text)

    # 替换 `\ne` 为 `≠`
    clean_text = re.sub(r'\\ne', '≠', clean_text)

    # 替换 \prime 为 ′
    clean_text = re.sub(r'\\prime', '′', clean_text)
    clean_text = re.sub(r'}′', '′', clean_text)
    clean_text = re.sub(r'{′', '′', clean_text)
    clean_text = re.sub(r'\^′', '′', clean_text)


    # 替换 `\frac{a}{b}` 为 `a/b`
    clean_text = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', clean_text)

    # 替换 `a{{x}^{2}}=b` 为 `a * x^2 = b`
    clean_text = re.sub(r'(\w+)\{\{(\w+)\}\^\{(\d+)\}\}', r'\1 * \2^\3', clean_text)

    # 替换乘号 `\cdot` 为 `*`
    clean_text = re.sub(r'\\cdot', '*', clean_text)

    # 替换 `\times` 为 `×`
    clean_text = re.sub(r'\\times', '×', clean_text)

    # 替换除法符号 `\div` 为 `/`
    clean_text = re.sub(r'\\div', '/', clean_text)

    # 替换 `\%` 为 `%`
    clean_text = re.sub(r'\\%', '%', clean_text)

    # 替换 `\sqrt{x}` 为 `sqrt(x)`
    clean_text = re.sub(r'\\sqrt\{(.+?)\}', r'sqrt(\1)', clean_text)

    # 替换幂次 `^{}` 为 `^`
    clean_text = re.sub(r'\^\{(\d+)\}', r'^\1', clean_text)

    # 替换小数点符号 `\dot{2}` 和 `\dot{1}` 为正常的小数点
    clean_text = re.sub(r'\\dot\{(\d+)\}', r'\1', clean_text)

    # 替换绝对值符号 `\left| \right|` 为 `| |`
    clean_text = re.sub(r'\\left\| (.+?) \\right\|', r'|\1|', clean_text)

    # 替换绝对值符号 `\left{ \right}` 为 `{ }`
    clean_text = re.sub(r'\\left\{ (.+?) \\right\}', r'{\1}', clean_text)

    # 替换括号 `\left( \right)` 为普通括号 `()`
    clean_text = re.sub(r'\\left\((.+?)\\right\)', r'(\1)', clean_text)

    # 替换不等式符号 `\leqslant` 和 `\geqslant` 为 `<=` 和 `>=`
    clean_text = re.sub(r'\\leqslant', '<=', clean_text)
    clean_text = re.sub(r'\\geqslant', '>=', clean_text)

    # 替换 `\triangle` 为 `△`
    clean_text = re.sub(r'\\triangle (\w+)', r'△\1',clean_text)

    clean_text = re.sub(r'\\begin\{cases\}', '条件表达式：', clean_text)
    clean_text = re.sub(r'\\end\{cases\}', '', clean_text)

    # 替换 LaTeX 中的 \text{km} 为 km
    clean_text = re.sub(r'\\text\{(\w+)\}', r'\1', clean_text)

    # 替换 `\begin{cases} ... \end{cases}` 为标准的不等式组表达
    clean_text = re.sub(r'\\begin\{cases\}(.+?)\\\\(.+?)\\end\{cases\}', r'{\1\n\2}', clean_text)

    # 去除无意义的 {} 符号
    clean_text = re.sub(r'\{(.+?)\}', r'\1', clean_text)

    # 替换 `\alpha` 为 `α`
    clean_text = re.sub(r'\\alpha', 'α', clean_text)

    # 替换 `\beta` 为 `β`
    clean_text = re.sub(r'\\beta', 'β',  clean_text)

    # 替换 \sqrt{x} 为 √(x)
    clean_text = re.sub(r'\\sqrt\{(.+?)\}', r'√(\1)', clean_text)
    # 替换 sqrt(x) 为 √(x)
    clean_text = re.sub(r'sqrt\((.+?)\)', r'√(\1)',  clean_text)
    clean_text = re.sub(r'\\sqrt(\d+)', r'√\1', clean_text)
    clean_text = re.sub(r'\\sqrt\s*\[\s*3\s*\]\s*-\s*(\d+)/(\d+)', r'∛-\frac{\1}{\2}', clean_text)

    # 替换 \sqrt[x]{y} 为 y^(1/x)
    clean_text = re.sub(r'\\sqrt\[(\d+)\]\{(.+?)\}', r'\2^(1/\1)',  clean_text)
    # 移除 LaTeX 数学环境符号 $
    clean_text =  clean_text.replace('$', '')
    # 移除 LaTeX 空白命令 \quad
    clean_text =  clean_text.replace('\\quad', '')
    clean_text = clean_text.replace('&nbsp;', ' ').replace('&there4;', '∴')


    return clean_text.strip()


def generate(model_path, label, test_data, outcome_path):
    # 定义不同类型的模型和tokenizer
    model_classes = [
        {"model_class": BloomForSequenceClassification, "tokenizer_class": AutoTokenizer,
         "model_name": model_path + '/bloom', "max_length": 512},
        {"model_class": XLNetForSequenceClassification, "tokenizer_class": XLNetTokenizer,
         "model_name": model_path + '/xlnet', "max_length": None},
        {"model_class": LlamaForSequenceClassification, "tokenizer_class": AutoTokenizer,
         "model_name": model_path + '/llama-3.2-1B', "max_length": 512},
        {"model_class": Qwen2ForSequenceClassification, "tokenizer_class": AutoTokenizer,
         "model_name": model_path + '/Qwen2.5', "max_length": 512},
        {"model_class": BertForSequenceClassification, "tokenizer_class": BertTokenizer,
         "model_name": model_path + '/chinese-roberta-wwm-ext', "max_length": 512},
        {"model_class": BertForSequenceClassification, "tokenizer_class": BertTokenizer,
         "model_name": model_path + '/Conan-embedding', "max_length": 512}
    ]
    models = []
    tokenizers = []
    max_lengths = []

    # 加载所有模型和tokenizer
    for item in model_classes:
        if item["model_class"] == LlamaForSequenceClassification:
            rope_scaling = {"type": "dynamic", "factor": 32}
            model = item["model_class"].from_pretrained(item["model_name"], rope_scaling=rope_scaling).to("cuda")
            model.eval()
            tokenizer = item["tokenizer_class"].from_pretrained(item["model_name"])
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 128004
            models.append(model)
            tokenizers.append(tokenizer)
            max_lengths.append(item["max_length"])  # 保存每个模型的 max_length
        elif item["model_class"] == Qwen2ForSequenceClassification:
            model = item["model_class"].from_pretrained(item["model_name"]).to("cuda")
            model.eval()
            tokenizer = item["tokenizer_class"].from_pretrained(item["model_name"])
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.pad_token_id = 151643
            models.append(model)
            tokenizers.append(tokenizer)
            max_lengths.append(item["max_length"])
        else:
            model = item["model_class"].from_pretrained(item["model_name"]).to("cuda")
            model.eval()
            tokenizer = item["tokenizer_class"].from_pretrained(item["model_name"])
            models.append(model)
            tokenizers.append(tokenizer)
            max_lengths.append(item["max_length"])  # 保存每个模型的 max_length

    # 题目字典  index+content
    data_ed = dict()
    # 答案字典   index+kn_id
    prompt_list = dict()
    # 错误字典
    error_list = dict()

    # 加载测试数据集
    with open(test_data, encoding="utf8") as f:
        list1 = csv.reader(f)
        for row in list1:
            row1 = row[2]
            prompt_list[row[1]] = row[5]  # 答案字典
            data = preprocess_text(row1)  # 处理数据
            data_ed[row[1]] = data

    # 结果字典
    res_completions = dict()
    count1 = 0
    all_predictions = []  # 用于保存所有预测结果
    all_labels = []  # 用于保存所有真实标签

    top_k = 2
    # 对测试数据进行处理
    for data in tqdm(data_ed.keys(), desc="Processing data"):
        prompt = data_ed[data]

        # 用于存储每个标签的累积概率
        label_scores = np.zeros(155)  # 假设有155个不同标签

        for model, tokenizer, max_length in zip(models, tokenizers, max_lengths):
            if max_length:
                inputs = tokenizer(prompt, truncation=True, padding=True, max_length=max_length,
                                   return_tensors="pt").to("cuda")
            else:
                inputs = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")  # XLNet不截断

            # 获取模型预测
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = softmax(logits, dim=-1).squeeze()  # 使用softmax获取概率分布

            # 获取Top-k的标签和对应的概率
            topk_probs, topk_indices = torch.topk(probs, top_k)

            # 将top-k结果的概率累积到label_scores中
            for i in range(top_k):
                label_scores[topk_indices[i].item()] += topk_probs[i].item()
              

                # 打印累加后的每个标签概率
        print(f"Data ID: {data}")
        print("Accumulated label probabilities:")
        for i, score in enumerate(label_scores):
            print(f"Label {i}: {score:.4f}")

        # 最终标签为累计概率最高的标签
        final_prediction = np.argmax(label_scores)
        if final_prediction in [23, 112] and ('顶点' in data_ed[data]):
            final_prediction = 140

        # 打印当前数据的预测和标签
        # print("data=", data_ed[data])
        # print(f"最终预测结果: {final_prediction}")
        # print(f"真实标签: {prompt_list[data]}")

        # 保存真实标签和预测结果
        all_predictions.append(final_prediction)
        all_labels.append(int(prompt_list[data]))

        # 检查预测是否正确
        # if int(prompt_list[data]) == final_prediction:
        #     print("预测正确")
        #     count1 += 1
        # else:
        #     print("预测错误")
        #     if prompt_list[data] in error_list:
        #         error_list[prompt_list[data]] = error_list[prompt_list[data]] + 1
        #     else:
        #         error_list[prompt_list[data]] = 1

        # 保存最终的预测结果
        res_completions[data] = final_prediction

    # 计算准确率
    acc = accuracy_score(all_labels, all_predictions)
    P = precision_score(all_labels, all_predictions, average='macro')
    R = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')  # 使用加权平均来计算 F1 值
    print(f"模型综合准确率: {acc}")
    print(f"模型P: {P}")
    print(f"模型R: {R}")
    print(f"F1 值: {f1}")
    # print(f"错误统计: {error_list}")

    with open(outcome_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入每一行数据
        for key, value in res_completions.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    print("Start to load test data.")
    # 测试集文件夹路径
    test_dir_path = sys.argv[1]
    # 结果存放路径
    result_csv = sys.argv[2]
    # 模型路径
    unzip_path = sys.argv[3]

    test_dir_path = os.path.join(test_dir_path, "test_3.csv")
    # 提交代码的解压路径
    label_path = os.path.join(unzip_path, "diaa_2_training_data_label.csv")
    model_path = os.path.join(unzip_path, "model")

    generate(model_path=model_path, label=label_path, test_data=test_dir_path, outcome_path=result_csv)
