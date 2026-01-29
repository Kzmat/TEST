# ScaleFL-med

Code for the following paper:

Fatih Ilhan, Gong Su and Ling Liu, "ScaleFL: Resource-Adaptive Federated Learning with Heterogeneous Clients," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, Canada, Jun. 18-22, 2023.

## Introduction

Federated learning (FL) is an attractive distributed learning paradigm supporting real-time continuous learning and client privacy by default. In most FL approaches, all edge clients are assumed to have sufficient computation capabilities to participate in the learning of a deep neural network (DNN) model. However, in real-life applications, some clients may have severely limited resources and can only train a much smaller local model. This paper presents ScaleFL, a novel FL approach with two distinctive mechanisms to handle resource heterogeneity and provide an equitable FL framework for all clients. First, ScaleFL adaptively scales down the DNN model along width and depth dimensions by leveraging early exits to find the best-fit models for resource-aware local training on distributed clients. In this way, ScaleFL provides an efficient balance of preserving basic and complex features in local model splits with various sizes for joint training while enabling fast inference for model deployment. Second, ScaleFL utilizes self-distillation among exit predictions during training to improve aggregation through knowledge transfer among subnetworks.
## Requirements
* Python 3.12
* PyTorch 2.7.0
* HuggingFace transformer 4.47
* HuggingFace datasets 3.6.0

## Usage example
python main.py \
    --data-root datasets/knee-X-ray \
    --data kneexray \
    --arch MiniGPTv2 \
    --use-valid \
    --cfg-path models/minigpt4/configs/models/minigpt_v2.yaml \
    --num_rounds 5 \
    --num_clients 4 \
    --sample_rate 1.0 \
    --batch-size 1
```



### Complete Parameters
```bash
python main.py \
    --data-root datasets/knee-X-ray \
    --data kneexray \
    --arch MiniGPTv2 \
    --use-valid \
    --cfg-path models/minigpt4/configs/models/minigpt_v2.yaml \
    --num_rounds 5 \
    --num_clients 4 \
    --sample_rate 1.0 \
    --batch-size 1 \
    --lr 3e-5 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --use_gpu
```