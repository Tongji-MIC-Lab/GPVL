##Source codes of GPVL

Requirements:
- Python 3.8
- pytorch = 1.9.1 cuda=11.1, torchvision==0.10.1, mmcv=0.14.0, torchaudio=0.9.1, mmdet==2.14.0, mmsegmentation==0.14.1
- Other python packages: apex, pycocotools, pyyaml, easydict, datasets,timm

## Data preparation
### make the prompts of detection, motion, map and global labels
python ./tools/det_motion_map_labels.py
### extract the detection, motion and map features
bash visual/extract.sh

## 3d-vision-language training
bash ./scripts/pretrain.sh

## trajectory finetuning
bash ./scripts/finetune_cap

## evaluation :
python test_by_pred_results.py