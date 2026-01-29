class Config:
    def __init__(self):
        self.training_params = {
            'rsna':{
                'MiniGPTv2': {
                'llama_model': 'models/minigpt4/llama-2-7b-chat-hf',
                'vit_model': 'eva_clip_g',
                'image_size': 224,
                'drop_path_rate': 0,
                'use_grad_checkpoint': False,
                'vit_precision': "fp16",
                'freeze_vit': True,

                'prompt': "",

                'image_path': 'datasets/RSNA/RSNA-bbox-1024',
                'ann_path': 'datasets/json_files/RSNA/RSNA_train.json',
                'vis_processor':"blip2_image_train",
                'text_processor':"blip_caption",

                # lora params
                'lora_r': 64,
                'lora_alpha': 16,
        
                'batch_size': 16,  
                'num_epoch': 5,  
                'lr': 3e-5,  
                'lr_type': 'none',  
                'weight_decay': 1e-4,  
                'momentum': 0.9,  
                'optimizer': 'sgd'  },
            },

            'kneexray':{
                'MiniGPTv2': {
                'llama_model': 'models/minigpt4/llama-2-7b-chat-hf',
                'vit_model': 'eva_clip_g',
                'image_size': 224,  # 减小图像尺寸以节省显存
                'drop_path_rate': 0,
                'use_grad_checkpoint': True,  # 启用梯度检查点
                'vit_precision': "fp16",
                'freeze_vit': True,

                'prompt': "",

                'image_path': 'datasets/knee-X-ray/image',
                'ann_path': 'datasets/knee-X-ray/knee_x_ray.json',
                'vis_processor':"blip2_image_train",
                'text_processor':"blip_caption",

                # lora params - 减少参数以节省显存
                'lora_r': 16,  # 从64减少到16
                'lora_alpha': 8,  # 从16减少到8
        
                'batch_size': 2,  # 从16减少到2
                'num_epoch': 5,  
                'lr': 3e-5,  
                'lr_type': 'none',  
                'weight_decay': 1e-4,  
                'momentum': 0.9,  
                'optimizer': 'sgd'  },
            },
            
            
            'cifar10': {
                'msdnet24_1': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 4e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'msdnet24_4': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 4e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'resnet110_1': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'},
                'resnet110_4': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'}
            },
            'cifar100': {
                'msdnet24_1': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 5e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'msdnet24_4': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 5e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'resnet110_1': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'},
                'resnet110_4': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'}
            },
            # lr details in submission for effnet were reported with some errors, we use exponential decay as suggested in the original paper.
            'imagenet': {
                'effnetb4_1': {'batch_size': 64,
                               'num_epoch': 5,
                               'lr': 0.2,
                               'lr_type': 'exp',
                               'decay_rate': 0.98,
                               'decay_epochs': [30, 60],
                               'weight_decay': 1e-5,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'effnetb4_4': {'batch_size': 64,
                               'num_epoch': 5,
                               'lr': 0.2,
                               'lr_type': 'exp',
                               'decay_rate': 0.98,
                               'decay_epochs': [30, 60],
                               'weight_decay': 1e-5,
                               'momentum': 0.9,
                               'optimizer': 'sgd'}
            },
            'sst2': {
                'bert_1': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                },
                'bert_4': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                }
            },
            'ag_news': {
                'bert_1': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                },
                'bert_4': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                }
            }
        }
        self.model_params = {
            'rsna':{
                'MiniGPTv2': {
                # vit encoder
                'vit_model': "eva_clip_g",
                'batch_size':6,
                'image_size': 224,
                'drop_path_rate': 0,
                'use_grad_checkpoint': False,
                'vit_precision': "fp16",
                'freeze_vit': True,


                'image_path': 'datasets/knee-X-ray/image',
                'ann_path': 'datasets/knee-X-ray/knee_x_ray.json',
                'vis_processor':"blip2_image_train",
                'text_processor':"blip_caption",

                'prompt': "",
                # llama model
                'llama_model': "models/minigpt4/llama-2-7b-chat-hf",

                # lora params
                'lora_r': 64,
                'lora_alpha': 16
                },
            },


            'kneexray':{
                'MiniGPTv2': {
                # vit encoder
                'vit_model': "eva_clip_g",
                'batch_size': 2,  # 从16减少到2
                'image_size': 224,  # 从448减少到224
                'drop_path_rate': 0,
                'use_grad_checkpoint': True,  # 启用梯度检查点
                'vit_precision': "fp16",
                'freeze_vit': True,


                'image_path': 'datasets/RSNA/RSNA-bbox-1024',
                'ann_path': 'datasets/json_files/RSNA/RSNA_train.json',
                'vis_processor':"blip2_image_train",
                'text_processor':"blip_caption",

                'prompt': "",
                # llama model
                'llama_model': "models/minigpt4/llama-2-7b-chat-hf",

                # lora params - 减少参数以节省显存
                'lora_r': 16,  # 从64减少到16
                'lora_alpha': 8  # 从16减少到8
                },
            },
            
            'cifar10': {
                'msdnet24_1': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 1,
                               'reduction': 0.5},
                'msdnet24_4': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 4,
                               'reduction': 0.5},
                'resnet110_1': {'ee_layer_locations': [],
                                'ee_num_conv_layers': [],
                                'num_blocks': 1},
                'resnet110_4': {'ee_layer_locations': [30, 38, 46],
                                'ee_num_conv_layers': [3, 3, 3],
                                'num_blocks': 4},
            },
            'cifar100': {
                'msdnet24_1': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 1,
                               'reduction': 0.5},
                'msdnet24_4': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 4,
                               'reduction': 0.5},
                'resnet110_1': {'ee_layer_locations': [],
                                'ee_num_conv_layers': [],
                                'num_blocks': 1},
                'resnet110_4': {'ee_layer_locations': [30, 38, 46],
                                'ee_num_conv_layers': [3, 3, 3],
                                'num_blocks': 4}
            },
            'imagenet': {
                'effnetb4_1': {'ee_layer_locations': [],
                               'ee_num_conv_layers': [],
                               'num_blocks': 1},
                'effnetb4_4': {'ee_layer_locations': [5, 6, 7],
                               'ee_num_conv_layers': [3, 3, 3],
                               'num_blocks': 4}
            },
            'sst2': {
                'bert_1': {'ee_layer_locations': [],
                           'num_blocks': 1},
                'bert_4': {'ee_layer_locations': [4, 6, 9],
                           'num_blocks': 4},
            },
            'ag_news': {
                'bert_1': {'ee_layer_locations': [],
                           'num_blocks': 1},
                'bert_4': {'ee_layer_locations': [4, 6, 9],
                           'num_blocks': 4},
            }
        }
