# AgonisticReplacementAdaTAD
Deployment of AdaTAD to detect agonistic replacement behaviour in cattle for an ethological case study. 

## How to run

### Training

You can train AdaTAD by running the tools/train.py script. 

```
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  tools/train.py \
  configs/adatad/displacement/e2e_displacement_2min_videomaev2_b_160x2_160_adapter_predecoded.py.py \
  --cfg-options dataset.train.empty_clip_ratio=0.0 dataset.val.empty_clip_ratio=0.0 work_dir=exps/displacement/adatad
```

### VideoMAEv2 model weights 

To run the training, the config file needs to contain the path to pretrained weights. 
VideoMAEv2 model weights fine-tuned on our trimmed clips can be downloaded from [here](https://drive.google.com/drive/folders/1fXBuSjbe_oR5hX6LRnkr1Mv0Qp9kNAaR?usp=sharing)

### Calibration for ethologically relevant thresholds

```
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/calibrate.py \
    configs/adatad/displacement/e2e_displacement_2min_videomaev2_b_160x2_160_adapter_predecoded.py \
    --checkpoint exps/displacement/adatad/path/to/checkpoint/best.pth \
    --tiou_match 0.3 \
    --target_recalls 0.6 0.7 0.8 0.9 \
    --nms_iou 0.3
```

### Testing

```
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/test.py \
    configs/adatad/displacement/e2e_displacement_2min_videomaev2_b_160x2_160_adapter_predecoded.py \
    --checkpoint exps/displacement/adatad/path/to/checkpoint/best.pth \
    --calibration exps/displacement/adatad/path/to/calibration.json \
    --operating_point max_f2 count_mae_optimal recall_80
```

## Acknowledgements
This work is built upon the [OpenTAD framework](https://github.com/sming256/OpenTAD).
