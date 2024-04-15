import multiprocessing
import warnings
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time

from data_utils import squarepad_transform, FashionIQDataset, ShoesDataset, Fashion200kDataset, targetpad_transform
from combiner import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device

def compute_fashion_val_metrics(relative_val_dataset: [ShoesDataset, Fashion200kDataset, Fashion200kDataset], clip_model: torch.nn.Module, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable, clip_preprocess) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fashion_val_predictions(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features, clip_preprocess)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())


    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at1, recall_at10, recall_at50


def generate_fashion_val_predictions(clip_model: torch.nn.Module, relative_val_dataset: [ShoesDataset, Fashion200kDataset, FashionIQDataset],
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor, clip_preprocess) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    pin_memory = True
    if isinstance(relative_val_dataset, Fashion200kDataset):
        pin_memory = False
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     pin_memory=pin_memory, collate_fn=collate_fn,
                                     shuffle=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, 512)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data
        if len(captions) == 2:
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
                i in range(0, len(flattened_captions), 2)]
        else:
            input_captions = captions
        text_embedding = clip_preprocess(text=input_captions, return_tensors="pt", max_length=77, padding="max_length", truncation=True).to(device)

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_embedding)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)
        predicted_features = torch.vstack((predicted_features, batch_predicted_features))
        target_names.extend(batch_target_names)

    return predicted_features, target_names

