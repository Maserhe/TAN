import os, warnings
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

from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import base_path, squarepad_transform, FashionIQDataset, Fashion200kDataset, ShoesDataset, targetpad_transform
from combiner import Combiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device, init_model
from validate import compute_fashion_val_metrics
from transformers import CLIPProcessor, CLIPModel
from albef_combiner import AlbefCombiner

def combiner_training_fashion(train_dress_types: List[str], val_dress_types: List[str], dataset: str,
                            num_workers: int,projection_dim: int, hidden_dim: int, num_epochs: int,
                            combiner_lr: float, batch_size: int, validation_frequency: int,
                            alpha: float, margin: float,momentum: float, save_best: bool, loss_id:[list], **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozed the CLIP model
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%m-%d_%H_%M_%S")
    training_path: Path = Path(
        base_path / f"models/{dataset}_model/base_mse_tri_{dataset}_lr{str(combiner_lr)}_margin{margin}_pro_dim{projection_dim}_hidden_dim{hidden_dim}_save{save_best}{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip", local_files_only=True).to(device=device).eval()
    text_preprocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip", local_files_only=True)

    input_dim = 224
    feature_dim = 512
    clip_preprocess = text_preprocess

    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIPModel"])
        print('CLIP model loaded successfully')

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []

    # Define the validation datasets and extract the validation index features for each dress_type
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
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    combiner_m = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)

    albef = AlbefCombiner(combiner, combiner_m, alpha=alpha, loss_ids=loss_id, margin=margin, momentum=momentum).to(device=device)


    pin_memory = True
    if train_dress_types[0] == 'shoes':
        relative_train_dataset = ShoesDataset('train', 'relative', clip_preprocess)
    elif dataset == "fashion200k":
        pin_memory = False
        relative_train_dataset = Fashion200kDataset('train', 'relative', clip_preprocess)
    else:
        relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', clip_preprocess)

    relative_train_loader = DataLoader(dataset=relative_train_dataset,
                                       batch_size=batch_size, pin_memory=pin_memory, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    crossentropy_criterion = nn.CrossEntropyLoss()
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
        relative_train_loader.dataset.epoch_count = epoch
        print(f"[{datetime.now()}] Training ...")
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(
                    train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                images_in_batch = reference_images['pixel_values'].size(0)
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
                # Extract the features with clip
                with torch.no_grad():
                    reference_image_features = clip_model.get_image_features(**reference_images).float()
                    target_image_features = clip_model.get_image_features(**target_images).float()
                    text_features = clip_model.get_text_features(**text_embedding).float()

                # Compute the logits and the loss
                with torch.cuda.amp.autocast():
                    logits, _, _ = combiner(reference_image_features, text_features, target_image_features)
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backprogate and update the weights
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
            print(f"[{datetime.now()}] Validating...")
            with experiment.validate():
                combiner.eval()
                recalls_at1 = []
                recalls_at10 = []
                recalls_at50 = []

                for relative_val_dataset, index_features, index_names, idx in zip(relative_val_datasets, index_features_list, index_names_list, idx_to_dress_mapping):
                    recall_at1, recall_at10, recall_at50 = compute_fashion_val_metrics(relative_val_dataset, clip_model, index_features, index_names, combiner.combine_features, text_preprocess)
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
                        f'average_recall': (mean(recalls_at1) + mean(recalls_at50) + mean(recalls_at10)) / 3
                    })
                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )
                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_best and results_dict['average_recall'] > best_avg_recall:
                best_avg_recall = results_dict['average_recall']
                save_model(dataset + '_combiner', epoch, combiner, base_path, optimizer)


if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--num_workers", type=int, required=False, default=0, help="For low memory consumption: limit cirr to 4 and fashioniq to 1 or 2. For optimal training efficiency, set to 8")
    # comet environment
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    # clip pretrain
    # training args
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--combiner-lr", default=2e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    # combiner training args
    parser.add_argument("--feature-dim", default=512, type=int, help="Feature dimension as input to combiner. Default: inherited from clip_model.visual.output_dim")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned clip model")
    # image preprocessing
    parser.add_argument("--input-dim", default=224, type=int, help="Input dimension for image transform. Default: inherited from clip_model.visual.input_resolution")
    # training settings
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    parser.add_argument("--loss-id", nargs='+', type=int, help="Save only the best model during training")
    parser.add_argument("--alpha", default=0.4, type=float, help="Save only the best model during training")
    parser.add_argument("--margin", default=0.2, type=float, help="Save only the best model during training")
    parser.add_argument("--momentum", default=0.995, type=float, help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'fashion200k', 'shoes']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "input_dim": args.input_dim,
        "feature_dim": args.feature_dim,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "num_workers": args.num_workers,
        "loss_id": args.loss_id,
        "alpha": args.alpha,
        "margin": args.margin,
        "momentum": args.momentum,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"fashionclip4Cir combiner_training {args.dataset}",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    random_seed = 0
    print(f"setting random seed to {random_seed}")
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
    combiner_training_fashion(**training_hyper_params)