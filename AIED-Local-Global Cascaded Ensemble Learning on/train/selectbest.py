import os
import shutil
import json
from glob import glob

dir1 = "chinese-roberta-wwm-ext"
# dir2 = "chinese-roberta-wwm-ext-large"
# dir3 = "chinese-roberta-wwm-ext-large2"
dir4 = "Conan-embedding"
# dir5 = "Conan-embedding2"
dir6 = "llama-3.2-1B"
dir7 = "bloom"
dir8 = "Qwen2.5"
dir9 = "xlnet"

# 比较两次训练的模型，选择较好的一个放入相应的目录
for dir in [dir1, dir4, dir6, dir7, dir8, dir9]:
    model1 = glob(f"./train-model/{dir}/**/trainer_state.json", recursive=True)
    # model2 = glob(f"./train-model/{dir}2/**/trainer_state.json", recursive=True)

    best_checkpoint = os.path.dirname(model1[0])

    for item in os.listdir(best_checkpoint):
        s = os.path.join(best_checkpoint, item)
        d = os.path.join(f"model/{dir}", item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

# 比较两次训练的模型，选择较好的一个放入相应的目录
# for dir in [dir1, dir2,dir3,dir4,dir5,dir6,dir7,dir8, dir9]:
#     model1 = glob(f"./train-model/{dir}/**/trainer_state.json", recursive=True)
#     model2 = glob(f"./train-model/{dir}2/**/trainer_state.json", recursive=True)
#
#     if model1 and model2:
#         with open(model1[0], 'r') as f:
#             metric1 = json.load(f).get('best_metric', 0)
#         with open(model2[0], 'r') as f:
#             metric2 = json.load(f).get('best_metric', 0)
#
#         if metric1 > metric2:
#             best_checkpoint = os.path.dirname(model1[0])
#         else:
#             best_checkpoint = os.path.dirname(model2[0])
#
#         for item in os.listdir(best_checkpoint):
#             s = os.path.join(best_checkpoint, item)
#             d = os.path.join(f"model/{dir}", item)
#             if os.path.isdir(s):
#                 shutil.copytree(s, d, dirs_exist_ok=True)
#             else:
#                 shutil.copy2(s, d)
#     else:
#         print(f"Trainer state file missing for {dir}")
#
# #把llama-3.2-1B的config.json中的rope_scaling改为None
# config_path = os.path.join("model", dir6, "config.json")
# if os.path.exists(config_path):
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#
#     config["rope_scaling"] = None
#
#     with open(config_path, 'w') as f:
#         json.dump(config, f, indent=4)