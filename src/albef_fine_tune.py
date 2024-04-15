import os
'''
Manually limiting the thread number for numpy
this is recommended if your CPU has many threads
'''
num_numpy_threads = '8'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import base_path, squarepad_transform, targetpad_transform, FashionIQDataset, ShoesDataset, Fashion200kDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device, cosine_lr_schedule, init_model
from validate import compute_fashion_val_metrics
from transformers import CLIPProcessor, CLIPModel
from albef import Albef

def clip_finetune_fashion(train_dress_types: List[str], val_dress_types: List[str],dataset: str,
                            num_epochs: int, batch_size: int, clip_learning_rate: float, clip_min_lr: float,
                            clip_max_epoch: int,validation_frequency: int, alpha: float, margin: float,momentum: float,
                            save_best: bool,
                            loss_id:[list], queue_size: int,
                            **kwargs):
    """
    Fine-tune clip text encoder on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param batch_size: batch size
    :param clip_learning_rate: fine-tuning learning rate
    :param clip_min_lr: minimum learning rate for cosine learning rate scheduler
    :param clip_max_epoch: maximum training epochs for cosine learning rate scheduler
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned clip model
    :param save_best: when True save only the weights of the best clip model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """
    grad_accumulation_step = 1 # gradient accumulation, though we have not used it

    training_start = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    training_path: Path = Path(
        base_path / f"models/finetune/base_mse_{dataset}_model/momentum{momentum}_margin{margin}_alpha{alpha}_lr_{str(clip_learning_rate)}_minLr{clip_min_lr}_epoch{num_epochs}_max_epoch{clip_max_epoch}_save_{save_best}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"training start time {training_start}")
    print(f"local folder {training_path}")

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, text_preprocess = init_model()
    clip_model_m, _ = init_model()
    clip_model.to(device=device).eval()
    clip_model_m.to(device=device).eval()

    albef = Albef(clip_model, clip_model_m, loss_ids=loss_id, alpha=alpha, margin=margin, momentum=momentum).to(device=device)


    clip_preprocess = text_preprocess
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        if dataset == "shoes":
            ShoesDataset.load_img_map()
            relative_val_dataset = ShoesDataset('val', 'relative', clip_preprocess)
            classic_val_dataset = ShoesDataset('val', 'classic', clip_preprocess)
        elif dataset == "fashion200k":
            Fashion200kDataset.load_img_map()
            relative_val_dataset = Fashion200kDataset('val', 'relative', clip_preprocess)
            classic_val_dataset = Fashion200kDataset('val', 'classic', clip_preprocess)
        else:
            FashionIQDataset.load_img_map()
            relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', clip_preprocess)
            classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', clip_preprocess)

        relative_val_datasets.append(relative_val_dataset)
        classic_val_datasets.append(classic_val_dataset)

    # Define the train datasets and the combining function
    if dataset == 'shoes':
        relative_train_dataset = ShoesDataset('train', 'relative', clip_preprocess)
    elif dataset == "fashion200k":
        relative_train_dataset = Fashion200kDataset('train', 'relative', clip_preprocess)
    else:
        relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', clip_preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,pin_memory=False, collate_fn=collate_fn, drop_last=True, shuffle=True)

    combining_function = element_wise_sum
    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': clip_learning_rate,
          'weight_decay': 0.05}])

    scaler = torch.cuda.amp.GradScaler()
    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        with experiment.train():
            clip_model.train()
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)
            cosine_lr_schedule(optimizer, epoch, clip_max_epoch, clip_learning_rate, clip_min_lr, onlyGroup0=True)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):
                images_in_batch = reference_images['pixel_values'].size(0)
                step = len(train_bar) * epoch + idx
                optimizer.zero_grad()
                reference_images = reference_images.to(device)
                target_images = target_images.to(device)

                if dataset == "fashioniq":
                    # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                    flattened_captions: list = np.array(captions).T.flatten().tolist()
                    input_captions = generate_randomized_fiq_caption(flattened_captions)
                else:
                    input_captions = captions
                text_embedding = text_preprocess(text=input_captions, return_tensors="pt", max_length=77,
                                                 padding="max_length", truncation=True).to(device)
                # Extract the features, compute the logits and the loss
                with torch.cuda.amp.autocast():
                    loss = albef(reference_images, target_images, text_embedding, combining_function)
                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                clip_model.eval()
                recalls_at1 = []
                recalls_at10 = []
                recalls_at50 = []

                for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                          idx_to_dress_mapping):

                    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
                    recall_at1, recall_at10, recall_at50 = compute_fashion_val_metrics(relative_val_dataset, clip_model,index_features,index_names, combining_function, text_preprocess)
                    recalls_at1.append(recall_at1)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    if dataset != "fashioniq":
                        results_dict[f'{idx_to_dress_mapping[i]}_recall_at1'] = recalls_at1[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]

                if dataset == "fashioniq":
                    results_dict.update({
                        f'average_recall_at10': mean(recalls_at10),
                        f'average_recall_at50': mean(recalls_at50),
                        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                    })
                else:
                    results_dict.update({
                        f'average_recall': (mean(recalls_at1) + mean(recalls_at10) + mean(recalls_at50)) / 3
                    })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(results_dict,epoch=epoch)

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_best and results_dict['average_recall'] > best_avg_recall:
                best_avg_recall = results_dict['average_recall']
                save_model(dataset + '_tuned_clip_best', epoch, clip_model, base_path, optimizer)


if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'shoes' or 'fashionIQ'")
    # comet environment
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    # clip pretrain
    parser.add_argument("--clip-pretrained-path", default='models/model_base.pth', type=str, help="path of the clip pretrained model weights")
    # training args
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-learning-rate", default=1e-5, type=float, help="clip text encoder learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    # cosine learning rate scheduler
    parser.add_argument("--clip-min-lr", default=0, type=float, help="Cos Learning Rate Scheduler min learning rate")
    parser.add_argument("--clip-max-epoch", default=10, type=int, help="Cos Learning Rate Scheduler max epoch")
    # image preprocessing
    parser.add_argument("--input-dim", default=224, type=int, help="Input dimension for image transform. Default: inherited from clip_model.visual.input_resolution")
    # training settings
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--save-best", dest="save_best", action='store_true',help="Save only the best model during training")

    parser.add_argument("--loss-ratio", default=0.1, type=float,
                        help="Save only the best model during training")
    parser.add_argument("--loss-id", nargs='+', type=int, help="Save only the best model during training")
    parser.add_argument("--queue-size", type=int, help="Save only the best model during training")
    parser.add_argument("--alpha", default=0.4, type=float, help="Save only the best model during training")
    parser.add_argument("--margin", default=0.2, type=float, help="Save only the best model during training")
    parser.add_argument("--momentum", default=0.995, type=float, help="Save only the best model during training")
    # momentum

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'shoes', 'fashion200k']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "clip_pretrained_path": args.clip_pretrained_path,
        "clip_learning_rate": args.clip_learning_rate,
        "clip_max_epoch": args.clip_max_epoch,
        "clip_min_lr": args.clip_min_lr,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "input_dim": args.input_dim,
        "save_best": args.save_best,
        "loss_ratio": args.loss_ratio,
        "loss_id": args.loss_id,
        "alpha": args.alpha,
        "margin": args.margin,
        "momentum": args.momentum,
        "queue_size": args.queue_size,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(api_key=args.api_key,project_name=f"clip4Cir_Bi clip_text_finetune {args.dataset}",workspace=args.workspace,disabled=False)
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(api_key="",project_name="",workspace="",disabled=True)

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)
    
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)

    if args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt'], 'dataset': 'fashioniq'})
    elif args.dataset.lower() == 'fashion200k':
        training_hyper_params.update(
            {'train_dress_types': ['fashion200k'], 'val_dress_types': ['fashion200k'], 'dataset': 'fashion200k'})
    elif args.dataset.lower() == 'shoes':
        training_hyper_params.update(
            {'train_dress_types': ['shoes'], 'val_dress_types': ['shoes'], 'dataset': 'shoes'})
    clip_finetune_fashion(**training_hyper_params)
