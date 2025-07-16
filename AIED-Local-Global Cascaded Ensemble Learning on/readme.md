# 项目复现说明


---

## 1. 模型下载

在开始复现前，请先下载所需的预训练模型权重及相关文件。

可前往以下平台下载：

- [Hugging Face 模型库](https://huggingface.co/models)   
- [魔搭 ModelScope](https://www.modelscope.cn/home)

请将模型文件夹按如下方式放置在项目根目录下的 `pretarin-model/` 目录中：

```
pretarin-model/
├── roberta/
├── conan-embedding/
├── xlnet/
├── bloom-1.1b/
├── llama-3.2-1b/
└── qwen2.5-1.5b/
```

---

## 2. 无监督训练

### 2.1 RoBERTa 和 Conan-Embedding

运行以下命令，使用 Masked Language Modeling (MLM) 方式进行无监督训练：

```bash
python unsupervised_MLM.py
```

### 2.2 XLNet、BLOOM-1.1B、Llama-3.2-1B 和 Qwen2.5-1.5B

运行以下命令，使用 Causal Language Modeling (CLM) 方式进行无监督训练：

```bash
python unsupervised_CLM.py
```

> 注意：请在代码中修改为相应模型的路径，例如：
> `model_path = "pretarin-model/bloom-1.1b"`

---

## 3. 监督训练

完成无监督训练后，运行以下脚本对各模型进行监督训练：

```bash
./run-model.sh
```

脚本会自动对所有模型进行微调训练，请确保脚本中的路径配置正确。

---

## 4. 模型预测

有监督训练完成后，运行以下命令进行预测：

```bash
python predict.py  ./ ./result.csv ./
```

---

## 附加说明

如果您想训练baseline,可以运行:
```bash
python train_baseline.py
```
> 注意：请在代码中修改为相应模型的路径，例如：
> `model_path = "pretarin-model/Deepseek-math-7B"`

如有问题欢迎提交 Issue 或联系维护者。