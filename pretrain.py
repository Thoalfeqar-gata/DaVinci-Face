import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import backbones.backbones as backbones
import datasets
import utility_scripts.face_recognition_heads as face_recognition_heads
import eval, os, time
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import lr_scheduler
from lightning.pytorch import loggers 
from augmentation import Augmenter
from configs.pretrain_davit_b_224_glint360k_adaface import config # change the config to change the script's behaviour
from multitask.face_recognition_subnet import FaceRecognitionEmbeddingSubnet



torch.set_float32_matmul_precision('medium')



class FaceRecognitionModel(pl.LightningModule):
    """
    PyTorch Lightning module for training a face recognition model.
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: contains all the hyperparameters.
        """
        super().__init__()
        self.framework = kwargs.get('framework')
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained = kwargs.get('pretrained')
        self.head_name = kwargs.get('head_name')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.num_classes = kwargs.get('num_classes')
        self.optimizer_name = kwargs.get('optimizer')
        self.weight_decay = kwargs.get('weight_decay')
        self.scheduler_name = kwargs.get('scheduler')
        self.milestones = kwargs.get('scheduler_milestones')
        self.start_factor = kwargs.get('start_factor')
        self.min_lr = kwargs.get('min_lr')
        self.warmup_epochs = kwargs.get('warmup_epochs')
        self.learning_rate = kwargs.get('learning_rate')
        self.margin = kwargs.get('margin')
        self.scale = kwargs.get('scale')
        self.h = kwargs.get('h')
        self.t_alpha = kwargs.get('t_alpha')
        self.max_epochs = kwargs.get('max_epochs')
        self.val_datasets = kwargs.get('val_datasets')
        self.save_hyperparameters()


        # Instantiate the backbone
        self.backbone = backbones.get_backbone(
            backbone_name=self.backbone_name,
        )
        state_dict = torch.load(os.path.join('data', 'models', 'pretrained models', 'davit_b_adaface_Glint360k', 'backbone.pth'), map_location='cpu')
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        state_dict = {k : v for k, v in state_dict.items() if not 'patch_embeds.0' in k} # remove the first patch embedding layer's weights
        self.backbone.load_state_dict(state_dict, strict = False)

        for name, param in self.backbone.named_parameters():
            if 'patch_embeds.0' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        


        # Instantiate the face recognition subnet
        feature_embedding_dim = self.backbone.get_multiscale_feature_dim()[-1][1] # the embedding dim from the last stage.
        
        self.recognition_subnet = FaceRecognitionEmbeddingSubnet(
            feature_embedding_dim=feature_embedding_dim,
            embedding_dim=self.embedding_dim
        )
        state_dict = torch.load(os.path.join('data', 'models', 'pretrained models', 'davit_b_adaface_Glint360k', 'recognition_subnet.pth'), map_location='cpu')
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.recognition_subnet.load_state_dict(state_dict, strict = False)
        self.recognition_subnet.requires_grad_(False)  # Ensure the recognition subnet is not trainable

        # Instantiate the margin head (AdaFace, ArcFace, or CosFace)
        self.margin_head = face_recognition_heads.build_head(
            head_type=self.head_name,
            embedding_size=self.embedding_dim,
            classnum=self.num_classes,
            m=self.margin,
            s=self.scale,
            h=self.h,
            t_alpha=self.t_alpha
        )
        
        state_dict = torch.load(os.path.join('data', 'models', 'pretrained models', 'davit_b_adaface_Glint360k', 'margin_head.pth'), map_location='cpu')
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.margin_head.load_state_dict(state_dict, strict = False)
        self.margin_head.requires_grad_(False)  # Ensure the margin head is not trainable
        

        self.backbone = torch.compile(self.backbone)
        self.recognition_subnet = torch.compile(self.recognition_subnet)
        self.margin_head = torch.compile(self.margin_head)
        
        self.global_counter = 0
        self.validation_outputs = []

    def forward(self, x):
        """
        Forward pass through the backbone.
        """
        multiscale_features = self.backbone(x)
        normalized_embedding, embedding_norm = self.recognition_subnet(multiscale_features)
        return normalized_embedding, embedding_norm

    def training_step(self, batch, batch_idx):
        """
        A single training step.
        """

        
        self.global_counter += 1
        
        images, labels = batch
        
        if self.backbone_name in ['convnext_v2', 'mobilenet']:
            images = images.to(memory_format = torch.channels_last)


        labels = labels['face_recognition']
        normalized_embedding, embedding_norm = self(images)
        
        output = self.margin_head(normalized_embedding, embedding_norm, labels)
        loss = F.cross_entropy(output, labels)
        accuracy = (output.argmax(dim=1) == labels).float().mean()

        self.log_dict({'train_loss': loss, 'train_acc': accuracy}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        A single validation step.
        """
        (image1_batch, image2_batch), label_batch = batch
        
        if self.backbone_name in ['convnext_v2', 'mobilenet']:
            image1_batch = image1_batch.to(memory_format = torch.channels_last)
            image2_batch = image2_batch.to(memory_format = torch.channels_last)
        
        # Get embeddings with tta
        embeddings1, norm1 = self(image1_batch)
        embeddings2, norm2 = self(image2_batch)
        flipped_embeddings1, flipped_norm1 = self(torch.flip(image1_batch, dims=[3]))
        flipped_embeddings2, flipped_norm2 = self(torch.flip(image2_batch, dims=[3]))

        #unnormalize
        embeddings1 = embeddings1 * norm1
        embeddings2 = embeddings2 * norm2
        flipped_embeddings1 = flipped_embeddings1 * flipped_norm1
        flipped_embeddings2 = flipped_embeddings2 * flipped_norm2

        # sum   
        embeddings1 = (embeddings1 + flipped_embeddings1)
        embeddings2 = (embeddings2 + flipped_embeddings2)

        # normalize
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)



        # Calculate distances
        diff = embeddings1 - embeddings2
        dist = torch.sum(torch.square(diff), axis=1)

        self.validation_outputs.append({'distances': dist, 'labels': label_batch, 'dataloader_idx': dataloader_idx})


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        
        outputs = self.validation_outputs
        # Group outputs by dataloader_idx
        grouped_outputs = [[] for _ in range(len(self.trainer.val_dataloaders))]
        for output in outputs:
            grouped_outputs[output['dataloader_idx']].append(output)

        metrics_to_log = {}
        for i, val_output in enumerate(grouped_outputs):
            if not val_output:
                continue
            distances = torch.cat([x['distances'] for x in val_output]).cpu().float().numpy()
            labels = torch.cat([x['labels'] for x in val_output]).cpu().float().numpy()
            
            dataset_name = self.val_datasets[i]
            
            metrics_results = eval.get_metrics_from_distances({dataset_name: (distances, labels)})
            
            mean_accuracy, mean_precision, mean_recall, mean_f1_score, global_auc_score, _, _, _ = metrics_results[dataset_name]

            metrics_to_log[f'{dataset_name}_val_acc'] = mean_accuracy

        self.log_dict(metrics_to_log, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_outputs.clear()


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        if self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay = self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate, weight_decay = self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")


        if self.scheduler_name == 'cosine':
            linear = lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor, end_factor=1.0, total_iters=self.warmup_epochs)
            cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min = self.min_lr)
            scheduler = lr_scheduler.SequentialLR(optimizer, [linear, cosine], milestones=[self.warmup_epochs])
        elif self.scheduler_name == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")


        return [optimizer], [scheduler]
    

    def on_train_epoch_start(self):
            """
            Called at the beginning of each training epoch.
            """
            # 1. Set everything to eval mode to disable DropPath and freeze BatchNorms
            self.backbone.eval()
            self.recognition_subnet.eval()
            self.margin_head.eval()
            
            # 2. Safely unfreeze the patch embedding layer, bypassing torch.compile wrapper issues
            for name, module in self.backbone.named_modules():
                # Matches "_orig_mod.patch_embeds.0" inside the compiled module
                if name.endswith('patch_embeds.0'):
                    module.train()


class FaceRecognitionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for face recognition datasets.
    """
    def __init__(self, **kwargs):
        """
        Args:
            dataset_name (str): The dataset name (e.g. VGGFace2_Dataset). 
            val_datasets (list): A list of dataset names for validation.
            batch_size (int): Batch size for the dataloaders.
            num_workers (int): Number of worker processes for data loading.
        """
        super().__init__()
        self.dataset_name = kwargs.get('dataset_name')
        self.val_datasets = kwargs.get('val_datasets')
        self.min_num_image_per_class = kwargs.get('min_num_images_per_class')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers')
        self.train_transform = kwargs.get('train_transform')
        self.val_transform = kwargs.get('val_transform')


    def setup(self, stage=None):
        """
        Set up the datasets.
        """
        # Training dataset
        dataset_class = getattr(datasets, self.dataset_name)
        self.train_dataset = dataset_class(transform=self.train_transform)
        self.num_classes = self.train_dataset.number_of_classes()
        
        # Validation datasets
        if self.val_datasets:
            self.instantiated_val_datasets = []
            for val_dataset_name in self.val_datasets:
                val_dataset_class = getattr(datasets, val_dataset_name)
                self.instantiated_val_datasets.append(val_dataset_class(transform=self.val_transform))


    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, pin_memory=True, prefetch_factor=3)

    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
        if not hasattr(self, 'instantiated_val_datasets') or not self.instantiated_val_datasets:
            return None
        
        val_loaders = []
        for val_dataset in self.instantiated_val_datasets:
            val_loaders.append(
                DataLoader(val_dataset, batch_size=self.batch_size, 
                                  num_workers=self.num_workers, pin_memory=True)
            )
        return val_loaders


def main(args):
    """
    Main training function.
    """
    pl.seed_everything(42)


    augmenter = Augmenter(crop_augmentation_prob=args['crop_prob'], low_res_augmentation_prob=args['low_res_prob'], photometric_augmentation_prob=args['photometric_prob'])
    image_size = 224
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        augmenter,
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_module = FaceRecognitionDataModule(train_transform=train_transform, val_transform=val_transform, **args)
    data_module.setup()
    
    model = FaceRecognitionModel(num_classes=data_module.num_classes, **args)
    
    checkpoint_path = os.path.join('checkpoints',f'{args['backbone_name']}_{args['head_name']}_{args['dataset_name']}')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok = True)

    steps_per_epoch = data_module.train_dataloader().dataset.__len__() // (args['batch_size'] * args['accumulate_grad_batches'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, # Uses the exact path without time.time()
        filename='{epoch}-{train_loss:.5f}',
        save_top_k=2,            # Save top 2 just to be safe
        save_last=True,          
        verbose=True,
        monitor='train_loss',
        mode='min',
        every_n_train_steps=int(steps_per_epoch // 4) 
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=loggers.TensorBoardLogger(save_dir=f'logs/{args["backbone_name"]}_{args["head_name"]}'),
        max_epochs=args['max_epochs'],
        accelerator='auto',
        precision=args['precision'],
        val_check_interval= 1/4, # Validate 4 times per epoch
        gradient_clip_val=args['gradient_clip_val'],
        gradient_clip_algorithm=args['gradient_clip_algorithm'],
        accumulate_grad_batches=args['accumulate_grad_batches']
    )

    last_checkpoint = os.path.join(checkpoint_path, 'last.ckpt')

    if os.path.exists(last_checkpoint):
        print(f"\n[INFO] Resuming training from: {last_checkpoint}\n")
        trainer.fit(model, data_module, ckpt_path=last_checkpoint)
    else:
        print("\n[INFO] No previous checkpoint found. Starting from scratch.\n")
        trainer.fit(model, data_module)


    # Save after training    
    output_path = os.path.join('data', 'models', 'pretrained models', f'{args['backbone_name']}_{args['head_name']}_{args['dataset_name']}')
    os.makedirs(output_path, exist_ok=True)
    
    torch.save(
        model.backbone.state_dict(),
        os.path.join(output_path, 'backbone.pth')
    )

    torch.save(
        model.recognition_subnet.state_dict(),
        os.path.join(output_path, 'recognition_subnet.pth')
    )

    torch.save(
        model.margin_head.state_dict(),
        os.path.join(output_path, 'margin_head.pth')
    )


if __name__ == '__main__':
    # Print the config to make sure it is correct
    for key, value in config.items():
        print(f'{key}: {value}')

    main(config)