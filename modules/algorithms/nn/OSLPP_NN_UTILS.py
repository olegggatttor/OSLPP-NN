import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules.algorithms.base.OSLPP import Params, do_l2_normalization, do_pca, select_closed_set_pseudo_labels, \
    evaluate_T
from modules.loaders.balanced_sampling import create_train_dataloader
from modules.loaders.common import set_seed, create_datasets
from modules.loaders.osda import create_datasets_sub, _load_tensors_sub
from modules.loaders.resnet_features.features_dataset import FeaturesDataset
from modules.logging.logger import MetricsLogger
from modules.scoring.metrics import get_labels_single_model, get_entropy_scores, get_margin_scores
from modules.scoring.metrics_ensemble import get_labels_multi_models
from modules.selection.uncertanties import select_initial_rejected, get_new_rejected
import gc
from copy import deepcopy


def train_NN(feats, labels, num_epochs, params, balanced, lr, get_model, initial=False):
    num_src_classes = params.num_common + params.num_src_priv

    if not initial:
        feats, labels = feats[labels >= 0], labels[labels >= 0]
        feats, labels = feats[labels < num_src_classes], labels[labels < num_src_classes]

    assert (labels.unique() == torch.arange(num_src_classes)).all()

    model = get_model(params, num_src_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().cuda()
    ds = FeaturesDataset(feats, labels)
    dl = create_train_dataloader(ds, 32, balanced)
    for ep in range(num_epochs):
        for f, l in dl:
            optimizer.zero_grad()
            loss_fn(model(f.cuda()), l.cuda()).backward()
            optimizer.step()
    return model.eval()


def predict_NN(model, feats, labels):
    ds = FeaturesDataset(feats, labels)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    model = model.eval()
    predictions = []
    with torch.no_grad():
        for f, l in dl:
            out = model(f.cuda())
            predictions.append(F.softmax(out, dim=1).detach().cpu())
    predictions = torch.cat(predictions, dim=0)
    return predictions


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -probs * probs.log()
    entropy = entropy.sum(dim=1)
    return entropy.mean()


def score_augmentations_test(params, models, predictions_T_2, transform, common, tgt_private, ini_S, logger, seed, t):
    with torch.no_grad():
        assert len(models) == 1

        preds_not_aug_T = predictions_T_2[0]

        aug_features, aug_labels = _load_tensors_sub(f'DomainNet_DCC_aug/{transform.name}',
                                                     params.target,
                                                     'test_aug', True)

        aug_features, aug_labels = torch.tensor(aug_features), torch.tensor(aug_labels)

        idxs = np.isin(aug_labels, common) + np.isin(aug_labels, tgt_private)

        aug_features, aug_labels = aug_features[idxs], aug_labels[idxs]

        cpy_S, aug_features = do_l2_normalization(deepcopy(ini_S), aug_features)
        cpy_S, aug_features = do_pca(cpy_S, aug_features, params.pca_dim)
        cpy_S, aug_features = do_l2_normalization(cpy_S, aug_features)

        preds_aug = predict_NN(models[0], aug_features, aug_labels)

        idxs_common = np.isin(aug_labels, common)
        idxs_private = np.isin(aug_labels, tgt_private)

        aug_common_preds, aug_common_labels, no_aug_common_preds = preds_aug[idxs_common], \
                                                                   aug_labels[idxs_common], \
                                                                   preds_not_aug_T[idxs_common]
        aug_private_preds, aug_private_labels, no_aug_private_preds = preds_aug[idxs_private], \
                                                                      aug_labels[idxs_private], \
                                                                      preds_not_aug_T[idxs_private]

        ce_loss = nn.CrossEntropyLoss()

        similarity_common = (ce_loss(aug_common_preds, no_aug_common_preds).detach().item() +
                             ce_loss(no_aug_common_preds, aug_common_preds).detach().item()) / 2

        similarity_private = (ce_loss(aug_private_preds, no_aug_private_preds).detach().item() +
                              ce_loss(no_aug_private_preds, aug_private_preds).detach().item()) / 2

        no_aug_common_conf = no_aug_common_preds.max(dim=1)[0].numpy().mean()
        no_aug_common_inv_entropy = get_entropy_scores(no_aug_common_preds).mean()
        no_aug_common_margin = get_margin_scores(no_aug_common_preds).mean()

        no_aug_private_conf = no_aug_private_preds.max(dim=1)[0].numpy().mean()
        no_aug_private_inv_entropy = get_entropy_scores(no_aug_private_preds).mean()
        no_aug_private_margin = get_margin_scores(no_aug_private_preds).mean()

        aug_common_conf = np.quantile(aug_common_preds.max(dim=1)[0].numpy(), 0.5)
        aug_common_inv_entropy = np.quantile(get_entropy_scores(aug_common_preds), 0.5)
        aug_common_margin = np.quantile(get_margin_scores(aug_common_preds), 0.5)

        aug_private_conf = np.quantile(aug_private_preds.max(dim=1)[0].numpy(), 0.5)
        aug_private_inv_entropy = np.quantile(get_entropy_scores(aug_private_preds), 0.5)
        aug_private_margin = np.quantile(get_margin_scores(aug_private_preds), 0.5)

        logger.augmentation_save(params.source, params.target, seed, t, transform,
                                 similarity_common, similarity_private,
                                 aug_common_conf, aug_common_inv_entropy, aug_common_margin,
                                 aug_private_conf, aug_private_inv_entropy, aug_private_margin,
                                 no_aug_common_conf, no_aug_common_inv_entropy, no_aug_common_margin,
                                 no_aug_private_conf, no_aug_private_inv_entropy, no_aug_private_margin,
                                 )


def train(params: Params, lr, epochs, num_models, select_reject_mode, seed, common, tgt_private,
          logger: MetricsLogger,
          balanced_config=lambda feats, common: (None, False),
          get_model=lambda params, num_src_classes: nn.Sequential(nn.Linear(params.pca_dim, params.proj_dim),
                                                                       nn.ReLU(),
                                                                       nn.Linear(params.proj_dim,
                                                                                 num_src_classes)).cuda().train(),
          weights=None, tops=None,
          osda=True, is_images=False,
          augmentation_test=False, transform=None,
          need_aug_features=False):
    set_seed(seed)
    print(params.source, '->', params.target, 'lr=', lr, 'seed=', seed)

    get_labels = get_labels_single_model if num_models == 1 else get_labels_multi_models
    if osda:
        (feats_S, labels_S), (feats_T, labels_T) = create_datasets_sub(params.dataset,
                                                                       params.source,
                                                                       params.target,
                                                                       common,
                                                                       tgt_private,
                                                                       transform,
                                                                       is_images=is_images,
                                                                       augmentation_test=
                                                                       augmentation_test or need_aug_features)
    else:
        (feats_S, labels_S), (feats_T, labels_T) = create_datasets(params.dataset,
                                                                   params.source,
                                                                   params.target,
                                                                   params.num_common,
                                                                   params.num_src_priv,
                                                                   params.num_tgt_priv)

    params.n_r = int(len(labels_T) * params.n_r)
    num_src_classes = params.num_common + params.num_src_priv

    ini_S = feats_S

    if not is_images:
        feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
        feats_S, feats_T = do_pca(feats_S, feats_T, params.pca_dim)
        feats_S, feats_T = do_l2_normalization(feats_S, feats_T)

    rejected = np.zeros((len(feats_T),), dtype=np.int)
    pseudo_labels = -np.ones((len(feats_T),), dtype=np.int)
    cs_pseudo_labels = pseudo_labels

    # initial
    feats_S, feats_T = torch.tensor(feats_S), torch.tensor(feats_T)
    labels_S, labels_T = torch.tensor(labels_S), torch.tensor(labels_T)
    feats_all = torch.cat((feats_S, feats_T), dim=0)

    uniform_ratio, balanced = balanced_config(feats_T, common)

    if need_aug_features:
        aug_features, aug_labels = _load_tensors_sub(f'DomainNet_DCC_aug/{transform.name}',
                                                     params.target,
                                                     'test_aug', augmentation_test or need_aug_features)

        aug_features, aug_labels = torch.tensor(aug_features), torch.tensor(aug_labels)

        idxs = np.isin(aug_labels, common) + np.isin(aug_labels, tgt_private)

        aug_features, aug_labels = aug_features[idxs], aug_labels[idxs]

        cpy_S, aug_features = do_l2_normalization(deepcopy(ini_S), aug_features)
        cpy_S, aug_features = do_pca(cpy_S, aug_features, params.pca_dim)
        _, aug_features = do_l2_normalization(cpy_S, aug_features)
        aug_features = torch.tensor(aug_features)
    else:
        aug_features = None

    for t in range(1, params.T):
        if t == 1:
            models = [train_NN(feats_S, labels_S, epochs, params,
                               get_model=get_model, balanced=True, lr=lr, initial=True) for _ in range(num_models)]
        else:
            models = [train_NN(feats_all, torch.cat([labels_S, pseudo_labels], dim=0), epochs, params,
                               get_model=get_model, balanced=True, lr=lr) for _ in range(num_models)]
        predictions_T_2 = torch.stack([predict_NN(model, feats_T, labels_T) for model in models], dim=0)

        print('Reserved:', torch.cuda.memory_reserved(0))

        if augmentation_test:
            score_augmentations_test(params, models, predictions_T_2, transform,
                                     common, tgt_private, ini_S, logger, seed, t)
        if need_aug_features:
            aug_preds = predict_NN(models[0], aug_features, labels_T)
        else:
            aug_preds = None
        cs_pseudo_labels = get_labels(predictions_T_2)
        selected = torch.tensor(
            select_closed_set_pseudo_labels(cs_pseudo_labels.numpy(), predictions_T_2,
                                            t, params.T,
                                            mode=select_reject_mode,
                                            uniform_ratio=uniform_ratio, balanced=balanced,
                                            weights=weights, tops=tops, aug_preds=aug_preds))

        if t == 1:
            rejected = torch.tensor(select_initial_rejected(predictions_T_2, params.n_r,
                                                            mode=select_reject_mode, weights=weights,
                                                            tops=tops, aug_preds=aug_preds))
        else:
            selected = selected * (1 - rejected)
            rejected_new = get_new_rejected(predictions_T_2, selected, rejected, mode=select_reject_mode,
                                            weights=weights, tops=tops, aug_preds=aug_preds)
            rejected[rejected_new == 1] = 1
        selected = selected * (1 - rejected)

        pseudo_labels = cs_pseudo_labels.clone()
        pseudo_labels[rejected == 1] = num_src_classes
        pseudo_labels[(rejected == 0) * (selected == 0)] = -1

        metrics = evaluate_T(params.num_common, params.num_src_priv, params.num_tgt_priv,
                             labels_T.numpy(), cs_pseudo_labels.numpy(), rejected.numpy())
        where = torch.where((selected == 1) + (rejected == 1))[0]
        metrics_selected = evaluate_T(params.num_common, params.num_src_priv, params.num_tgt_priv,
                                      labels_T[where].numpy(), cs_pseudo_labels[where].numpy(),
                                      rejected[where].numpy())
        logger.log(t, metrics, metrics_selected)

        del predictions_T_2, models, metrics, metrics_selected, selected
        gc.collect()

    assert (pseudo_labels == -1).sum() == 0

    _rejected = rejected
    metrics = evaluate_T(params.num_common, params.num_src_priv, params.num_tgt_priv, labels_T.numpy(),
                         cs_pseudo_labels.numpy(), _rejected.numpy())
    logger.log_res(metrics)
    return metrics
