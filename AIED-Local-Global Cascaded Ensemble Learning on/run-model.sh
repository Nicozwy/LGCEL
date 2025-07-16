#!/bin/bash
export WANDB_MODE=disabled
# 在 model 文件夹中创建 6 个新目录
log_file="model_creation.log"
echo "Creating directories and moving files..." >> "$log_file"

dir1="chinese-roberta-wwm-ext"
#dir2="chinese-roberta-wwm-ext-large"
#dir3="chinese-roberta-wwm-ext-large2"
dir4="Conan-embedding"
#dir5="Conan-embedding2"
dir6="llama-3.2-1B"
dir7="bloom"
dir8="Qwen2.5"
dir9="xlnet"

mkdir -p "model/$dir1"
#mkdir -p "model/$dir2"
#mkdir -p "model/$dir3"
mkdir -p "model/$dir4"
#mkdir -p "model/$dir5"
mkdir -p "model/$dir6"
mkdir -p "model/$dir7"
mkdir -p "model/$dir8"
mkdir -p "model/$dir9"

echo "done mkdir" >> "$log_file"

echo "moving files..." >> "$log_file"
# 将预训练的模型文件（除了 config.json 和 pytorch_model.bin）移动到相应的目录
for dir in pretrain-model/*; do
    if [ -d "$dir" ]; then
        case $(basename "$dir") in
            chinese-roberta-wwm-ext)
                target_dirs=("model/chinese-roberta-wwm-ext")
                ;;
#            chinese-roberta-wwm-ext-large)
#                target_dirs=("model/chinese-roberta-wwm-ext-large" "model/chinese-roberta-wwm-ext-large2")
#                ;;
            Conan-embedding)
                target_dirs=("model/Conan-embedding")
                ;;
            llama-3.2-1B)
                target_dirs=("model/llama-3.2-1B")
                ;;
            bloom)
                target_dirs=("model/bloom")
                ;;
            Qwen2.5)
                target_dirs=("model/Qwen2.5")
                ;;
            xlnet)
                target_dirs=("model/xlnet")
                ;;
            *)
                echo "Unknown directory: $dir" >> "$log_file"
                continue
                ;;
        esac

        for file in "$dir"/*; do
            if [[ $(basename "$file") != "config.json" && $(basename "$file") != "pytorch_model.bin" ]]; then
                for target_dir in "${target_dirs[@]}"; do
                    cp -r "$file" "$target_dir/"
                done
            fi
        done
    fi
done

echo "done moving files" >> "$log_file"

#每个模型训练两次，最终选取最好的作为预测模型，参数为模型保存路径
echo "train model bert..." >> "$log_file"
python ./train/train-bert.py "train-model/$dir1" &>> "$log_file"
#python ./train/train-bert.py "train-model/${dir1}2" &>> "$log_file"
echo "done train model bert..." >> "$log_file"

#echo "train model bertlarge..." >> "$log_file"
#python ./train/train-bertlarge.py "train-model/$dir2" &>> "$log_file"
##python ./train/train-bertlarge.py "train-model/${dir2}2" &>> "$log_file"
#echo "done train model bertlarge..." >> "$log_file"

#echo "train model bertlarge2..." >> "$log_file"
#python ./train/train-bertlarge2.py "train-model/$dir3" &>> "$log_file"
#python ./train/train-bertlarge2.py "train-model/${dir3}2" &>> "$log_file"
#echo "done train model bertlarge2..." >> "$log_file"

echo "train model Conan-embedding..." >> "$log_file"
python ./train/train-Conan-embedding.py "train-model/$dir4" &>> "$log_file"
#python ./train/train-Conan-embedding.py "train-model/${dir4}2" &>> "$log_file"
echo "done train model Conan-embedding..." >> "$log_file"

#echo "train model Conan-embedding2..." >> "$log_file"
#python ./train/train-Conan-embedding2.py "train-model/$dir5" &>> "$log_file"
#python ./train/train-Conan-embedding2.py "train-model/${dir5}2" &>> "$log_file"
#echo "done train model Conan-embedding2..." >> "$log_file"

echo "train model llama..." >> "$log_file"
python ./train/train-llama.py "train-model/$dir6" &>> "$log_file"
#python ./train/train-llama.py "train-model/${dir6}2" &>> "$log_file"
echo "done train model llama..." >> "$log_file"

echo "train model bloom..." >> "$log_file"
python ./train/train-bloom.py "train-model/$dir7" &>> "$log_file"
#python ./train/train-bloom.py "train-model/${dir7}2" &>> "$log_file"
echo "done train model bloom..." >> "$log_file"

echo "train model Qwen..." >> "$log_file"
python ./train/train-Qwen.py "train-model/$dir8" &>> "$log_file"
#python ./train/train-Qwen.py "train-model/${dir8}2" &>> "$log_file"
echo "done train model Qwen..." >> "$log_file"

echo "train model xlnet..." >> "$log_file"
python ./train/train-xlnet.py "train-model/$dir9" &>> "$log_file"
#python ./train/train-xlnet.py "train-model/${dir9}2" &>> "$log_file"
echo "done train model xlnet..." >> "$log_file"

# 比较两次训练的模型，选择较好的一个放入model目录
python ./train/selectbest.py &>> "$log_file"
