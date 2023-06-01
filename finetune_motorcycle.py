import os
import json
import shutil
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.metainfo import Metrics, Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile
from modelscope.outputs import OutputKeys
from modelscope.hub.snapshot_download import snapshot_download

WORKSPACE = "./workspace"
image_model = 'damo/ofa_image-classification_imagenet_large_en'
image_path = snapshot_download(image_model) # 下载模型至缓存目录，并返回目录
# ofa通用的pretrained模型，未针对Image场景做过调优
pretrained_model = 'glt3953/motorcycle_brand' # 预训练模型的模型id
pretrained_path = snapshot_download(pretrained_model, revision='v1.0.0') # 预训练模型tag时间低于modelscope v1.0.2的发布时间，所以使用ms 1.0.2版本时需要额外增加具体的tag version
shutil.copy(os.path.join(image_path, ModelFile.CONFIGURATION), # 将任务的配置覆盖预训练模型的配置
            os.path.join(pretrained_path, ModelFile.CONFIGURATION))
os.makedirs(WORKSPACE, exist_ok=True)
config_file = os.path.join(WORKSPACE, ModelFile.CONFIGURATION) # 写一下配置文件
with open(config_file, 'w') as writer:
    json.dump(finetune_cfg, writer, indent="\t")
# trainer的其他配置项
args = dict(
    model=pretrained_path, # 要继续finetune的模型
    work_dir=WORKSPACE,
    train_dataset=MsDataset.load( # 数据集，这里msdataset兼容huggingface的dataset
        'glt3953/motorcycle_brand', # msdataset的id
        namespace='modelscope',
        split='train'),
    eval_dataset=MsDataset.load('glt3953/motorcycle_brand', namespace='modelscope', split='validation'),
    cfg_file=config_file) # 配置文件地址
trainer = build_trainer(name=Trainers.ofa, default_args=args) # 构建训练器
trainer.train()
