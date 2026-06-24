# AdaTAD

> [End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames](https://arxiv.org/abs/2311.17241)  
> Shuming Liu, Chen-Lin Zhang, Chen Zhao, Bernard Ghanem

## Abstract

Recently, temporal action detection (TAD) has seen significant performance improvement with end-to-end training. However, due to the memory bottleneck, only models with limited scales and limited data volumes can afford end-to-end training, which inevitably restricts TAD performance. In this paper, we reduce the memory consumption for end-to-end training, and manage to scale up the TAD backbone to 1 billion parameters and the input video to 1,536 frames, leading to significant detection performance. The key to our approach lies in our proposed temporal-informative adapter (TIA), which is a novel lightweight module that reduces training memory. Using TIA, we free the humongous backbone from learning to adapt to the TAD task by only updating the parameters in TIA. TIA also leads to better TAD representation by temporally aggregating context from adjacent frames throughout the backbone. We evaluate our model across four representative datasets. Owing to our efficient design, we are able to train end-to-end on VideoMAEv2-giant and achieve 75.4% mAP on THUMOS14, being the first end-to-end model to outperform the best feature-based methods.

## Prepare the pretrained VideoMAE checkpoints

Before running the experiments, please download the pretrained VideoMAE model weights (converted from original repo), and put them under the path `./pretrained/`.

- Note that we are not allowed to redistribute VideoMAEv2's checkpoints. You can fill out the official [request form], then convert the checkpoint by the following command.

```bash
python tools/model_converters/convert_videomaev2.py \
    vit_g_hybrid_pt_1200e_k710_ft.pth pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
```

## ActivityNet Results

Please refer to [README.md](../../tools/prepare_data/activitynet/README.md#download-raw-videos) to prepare the raw video of ActivityNet.

- To train the model on ActivityNet, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py
```

- To use the same checkpoint but test with another classifier, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py configs/adatad/anet/e2e_anet_videomaev2_g_192x4_224_adapter_internvideo2.py --checkpoint epoch_10_cba1017a.pth
```

**[NEW]** We provide the following checkpoints which does not require external classifier but directly trains 200 classification head, for the convenience of zero-shot inference.

## THUMOS-14 Results

Please refer to [README.md](../../tools/prepare_data/thumos/README.md#download-raw-videos) to prepare the raw video of THUMOS.

- To train the model on THUMOS, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py
```

- To search the adapter's learning rate, or change other hyper-parameters, you can run the following command.

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py \ 
  --cfg-options optimizer.backbone.custom.0.lr=1e-4 --id 1
```
