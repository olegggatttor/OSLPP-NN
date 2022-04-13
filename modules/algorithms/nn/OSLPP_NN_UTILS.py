import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules.algorithms.base.OSLPP import Params, do_l2_normalization, do_pca, select_closed_set_pseudo_labels, \
    evaluate_T
from modules.loaders.balanced_sampling import create_train_dataloader
from modules.loaders.common import set_seed
from modules.loaders.osda import create_datasets_sub
from modules.loaders.resnet_features.features_dataset import FeaturesDataset
from modules.selection.uncertanties import select_initial_rejected, get_new_rejected


def train_NN(feats, labels, num_epochs, params, balanced, lr, initial=False):
    num_src_classes = params.num_common + params.num_src_priv
    assert (labels.unique() == torch.arange(num_src_classes)).all()

    if not initial:
        feats, labels = feats[labels >= 0], labels[labels >= 0]
        feats, labels = feats[labels < num_src_classes], labels[labels < num_src_classes]

    model = nn.Sequential(nn.Linear(params.pca_dim, params.proj_dim), nn.ReLU(),
                          nn.Linear(params.proj_dim, num_src_classes)).cuda().train()
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
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    model = model.eval()
    feats = []
    predictions = []
    with torch.no_grad():
        for f, l in dl:
            out = model(f.cuda())
            predictions.append(F.softmax(out, dim=1).detach().cpu())
            feats.append(f.detach())
    feats = torch.cat(feats, dim=0)
    predictions = torch.cat(predictions, dim=0)
    return feats, predictions


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -probs * probs.log()
    entropy = entropy.sum(dim=1)
    return entropy.mean()


def train_osda(params: Params, lr, epochs, select_reject_mode, seed, common, tgt_private, logger,
               balanced_config=lambda feats, common: (None, False)):
    set_seed(seed)
    print(params.source, '->', params.target, 'lr=', lr, 'seed=', seed)
    (feats_S, labels_S), (feats_T, labels_T) = create_datasets_sub(params.dataset,
                                                                   params.source,
                                                                   params.target,
                                                                   common,
                                                                   tgt_private)
    # params.n_r = int(len(labels_T) * n_r)
    num_src_classes = params.num_common + params.num_src_priv

    # l2 normalization and pca
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
    feats_S, feats_T = do_pca(feats_S, feats_T, params.pca_dim)
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)

    # initial
    feats_S, feats_T = torch.tensor(feats_S), torch.tensor(feats_T)
    labels_S, labels_T = torch.tensor(labels_S), torch.tensor(labels_T)
    feats_all = torch.cat((feats_S, feats_T), dim=0)

    uniform_ratio, balanced = balanced_config(feats_T, common)

    t = 1

    model = train_NN(feats_S, labels_S, epochs, params, balanced=True, lr=lr, initial=True)
    feats_S_2, predictions_S_2 = predict_NN(model, feats_S, labels_S)
    feats_T_2, predictions_T_2 = predict_NN(model, feats_T, labels_T)

    assert (feats_S_2 == feats_S).all() and (feats_T_2 == feats_T).all()

    confs, cs_pseudo_labels = predictions_T_2.max(dim=1)
    selected = torch.tensor(select_closed_set_pseudo_labels(cs_pseudo_labels.numpy(), confs.numpy(), predictions_T_2,
                                                            t, params.T,
                                                            mode=select_reject_mode,
                                                            uniform_ratio=uniform_ratio, balanced=balanced))
    rejected = torch.tensor(select_initial_rejected(confs, predictions_T_2, params.n_r, mode=select_reject_mode))
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

    for t in range(2, params.T):
        model = train_NN(feats_all, torch.cat([labels_S, pseudo_labels], axis=0), epochs, params,
                         balanced=True, lr=lr)
        feats_S_2, predictions_S_2 = predict_NN(model, feats_S, labels_S)
        feats_T_2, predictions_T_2 = predict_NN(model, feats_T, labels_T)
        assert (feats_S_2 == feats_S).all() and (feats_T_2 == feats_T).all()

        confs, cs_pseudo_labels = predictions_T_2.max(dim=1)
        selected = torch.tensor(
            select_closed_set_pseudo_labels(cs_pseudo_labels.numpy(), confs.numpy(), t, params.T,
                                            predictions_T_2, select_reject_mode,
                                            uniform_ratio=uniform_ratio, balanced=balanced))

        selected = selected * (1 - rejected)
        rejected_new = get_new_rejected(confs, predictions_T_2, selected, rejected, mode=select_reject_mode)
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

    assert (pseudo_labels == -1).sum() == 0

    _rejected = rejected
    metrics = evaluate_T(params.num_common, params.num_src_priv, params.num_tgt_priv, labels_T.numpy(),
                         cs_pseudo_labels.numpy(), _rejected.numpy())
    logger.log_res(metrics)
    return metrics
