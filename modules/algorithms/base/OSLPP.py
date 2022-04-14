from sklearn.decomposition import PCA
import scipy
import numpy as np
import scipy.io
import scipy.linalg
from dataclasses import dataclass
from typing import Optional

from modules.loaders.osda import create_datasets_sub
from modules.scoring.metrics import get_acc_i, get_acc_c, get_h_score, mean
from modules.selection.uncertanties import select_closed_set_pseudo_labels_by_mode, SelectRejectMode, \
    select_initial_rejected


def get_num_classes(num_common, num_src_priv, num_tgt_priv):
    num_src = num_common + num_src_priv
    num_total = num_src + num_tgt_priv
    return num_src, num_total


def get_l2_norm(features: np.ndarray):
    return np.sqrt(np.square(features).sum(axis=1)).reshape((-1, 1))


def get_l2_normalized(features: np.ndarray): return features / get_l2_norm(features)


def get_PCA(features, dim):
    result = PCA(n_components=dim).fit_transform(features)
    assert len(features) == len(result)
    return result


def get_W(labels, ):
    W = (labels.reshape(-1, 1) == labels).astype(np.int)
    negative_one_idxs = np.where(labels == -1)[0]
    W[:, negative_one_idxs] = 0
    W[negative_one_idxs, :] = 0
    return W


def get_D(W): return np.eye(len(W), dtype=np.int) * W.sum(axis=1)


def fix_numerical_assymetry(M): return (M + M.transpose()) * 0.5


def get_projection_matrix(features, labels, proj_dim):
    N, d = features.shape
    X = features.transpose()

    W = get_W(labels)
    D = get_D(W)
    L = D - W

    A = fix_numerical_assymetry(np.matmul(np.matmul(X, D), X.transpose()))
    B = fix_numerical_assymetry(np.matmul(np.matmul(X, L), X.transpose()) + np.eye(d))
    assert (A.transpose() == A).all() and (B.transpose() == B).all()

    w, v = scipy.linalg.eigh(A, B)
    assert w[0] < w[-1]
    w, v = w[-proj_dim:], v[:, -proj_dim:]
    assert np.abs(np.matmul(A, v) - w * np.matmul(B, v)).max() < 1e-5

    w = np.flip(w)
    v = np.flip(v, axis=1)

    for i in range(v.shape[1]):
        if v[0, i] < 0:
            v[:, i] *= -1
    return v


def project_features(P, features):
    # P: pca_dim x proj_dim
    # features: N x pca_dim
    # result: N x proj_dim
    return np.matmul(P.transpose(), features.transpose()).transpose()


def get_centroids(features, labels):
    centroids = np.stack([features[labels == c].mean(axis=0) for c in np.unique(labels)], axis=0)
    centroids = get_l2_normalized(centroids)
    return centroids


def get_dist(f, features):
    return get_l2_norm(f - features)


def get_closed_set_pseudo_labels(features_S, labels_S, features_T):
    centroids = get_centroids(features_S, labels_S)
    dists = np.stack([get_dist(f, centroids)[:, 0] for f in features_T], axis=0)
    pseudo_labels = np.argmin(dists, axis=1)
    pseudo_probs = np.exp(-dists[np.arange(len(dists)), pseudo_labels]) / np.exp(-dists).sum(axis=1)
    return pseudo_labels, pseudo_probs


def select_closed_set_pseudo_labels(pseudo_labels, pseudo_probs, predictions, t, T, mode, uniform_ratio=15,
                                    balanced=False, weights=None, tops=None):
    if t >= T:
        t = T - 1
    selected = np.zeros_like(pseudo_labels)
    for c in np.unique(pseudo_labels):
        class_indices = np.where(pseudo_labels == c)[0]
        Nc = len(class_indices)
        if Nc > 0:
            selected = select_closed_set_pseudo_labels_by_mode(
                class_indices, selected, pseudo_probs, predictions,
                t, T, Nc,
                mode,
                uniform_ratio, balanced, weights=weights, tops=tops)
    return selected


def update_rejected(selected, rejected, features_T):
    unlabeled = (selected == 0) * (rejected == 0)
    new_is_rejected = rejected.copy()
    for idx in np.where(unlabeled)[0]:
        dist_to_selected = get_dist(features_T[idx], features_T[selected == 1]).min()
        dist_to_rejected = get_dist(features_T[idx], features_T[rejected == 1]).min()
        if dist_to_rejected < dist_to_selected:
            new_is_rejected[idx] = 1
    return new_is_rejected


def _fix_labels_with_value(original, value, where):
    result = original.copy()
    result[where] = value
    return result


def evaluate_T(num_common, num_src_priv, num_tgt_priv, labels, cs_pseudo_labels, rejected):
    num_known_classes = num_common + num_src_priv
    classes_preds = cs_pseudo_labels
    lbls = labels
    where_knw = lbls < num_common
    # closed set
    acc_i = get_acc_i(lbls[where_knw], classes_preds[where_knw])
    acc_c = get_acc_c(lbls[where_knw], classes_preds[where_knw], list(range(num_common)))
    # open set
    recall_knw = mean((1 - rejected)[where_knw])
    recall_unk = mean(rejected[~where_knw])
    # total
    tgt_classes = list(range(num_common))
    if num_tgt_priv > 0: tgt_classes += [num_known_classes]

    classes_preds = _fix_labels_with_value(classes_preds, num_known_classes, (rejected == 1))
    total_acc_i = get_acc_i(lbls, classes_preds)
    total_acc_c = get_acc_c(lbls, classes_preds, tgt_classes)
    total_h_score = get_h_score(lbls, classes_preds, list(range(num_common)), num_known_classes)
    return {'cs/acc_i': acc_i, 'cs/acc_c': acc_c,
            'os/recall_knw': recall_knw, 'os/recall_unk': recall_unk,
            'total/acc_i': total_acc_i, 'total/acc_c': total_acc_c, 'total/h_score': total_h_score}


def do_l2_normalization(feats_S, feats_T):
    feats_S, feats_T = get_l2_normalized(feats_S), get_l2_normalized(feats_T)
    assert np.abs(get_l2_norm(feats_S) - 1.).max() < 1e-5
    assert np.abs(get_l2_norm(feats_T) - 1.).max() < 1e-5
    return feats_S, feats_T


def do_pca(feats_S, feats_T, pca_dim):
    feats = np.concatenate([feats_S, feats_T], axis=0)
    feats = get_PCA(feats, pca_dim)
    feats_S, feats_T = feats[:len(feats_S)], feats[len(feats_S):]
    return feats_S, feats_T


def do_center(zs_S, zs_T):
    # center
    zs_mean = np.concatenate((zs_S, zs_T), axis=0).mean(axis=0).reshape((1, -1))
    zs_S = zs_S - zs_mean
    zs_T = zs_T - zs_mean
    return zs_S, zs_T


@dataclass
class Params:
    pca_dim: int  # = 512
    proj_dim: int  # = 128
    T: int  # = 10
    n_r: Optional[int]  # = 1200
    n_r_ratio: Optional[float]  # = 0.25
    dataset: str  # = 'OfficeHome'
    source: str  # = 'art'
    target: str  # = 'clipart'
    num_common: int  # = 25
    num_src_priv: int  # = 0
    num_tgt_priv: int  # = 40


def get_n_r(params: Params, lbls_T):
    if (params.n_r is not None) and (params.n_r_ratio is not None):
        raise Exception('Unsupported ambiguity: only 1 of n_r, n_r_ratio should be not None')
    if (params.n_r is not None):
        return params.n_r
    if params.n_r_ratio is not None:
        return int(len(lbls_T) * params.n_r_ratio)
    raise Exception('Unsupported ambiguity: 1 of n_r, n_r_ratio should be not None')


def main(params: Params, commons, tgt_privates):
    select_reject_mode = SelectRejectMode.CONFIDENCE
    (feats_S, lbls_S), (feats_T, lbls_T) = create_datasets_sub(params.dataset,
                                                               params.source,
                                                               params.target,
                                                               commons,
                                                               tgt_privates)

    # l2 normalization and pca
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)
    feats_S, feats_T = do_pca(feats_S, feats_T, params.pca_dim)
    feats_S, feats_T = do_l2_normalization(feats_S, feats_T)

    # initial
    feats_all = np.concatenate((feats_S, feats_T), axis=0)
    pseudo_labels = -np.ones_like(lbls_T)
    cs_pseudo_labels = pseudo_labels
    rejected = np.zeros_like(pseudo_labels)

    # iterations
    for t in range(1, params.T + 1):
        P = get_projection_matrix(feats_all, np.concatenate((lbls_S, pseudo_labels), axis=0), params.proj_dim)
        proj_S, proj_T = project_features(P, feats_S), project_features(P, feats_T)
        proj_S, proj_T = do_center(proj_S, proj_T)
        proj_S, proj_T = do_l2_normalization(proj_S, proj_T)

        cs_pseudo_labels, cs_pseudo_probs = get_closed_set_pseudo_labels(proj_S, lbls_S, proj_T)
        selected = select_closed_set_pseudo_labels(cs_pseudo_labels, cs_pseudo_probs, t, params.T, None,
                                                   select_reject_mode)
        selected = selected * (1 - rejected)

        if t == 2:
            rejected = select_initial_rejected(cs_pseudo_probs, None, get_n_r(params, lbls_T), mode=select_reject_mode)
        if t >= 2:
            rejected = update_rejected(selected, rejected, proj_T)
        selected = selected * (1 - rejected)

        pseudo_labels = cs_pseudo_labels.copy()
        if t < params.T:
            pseudo_labels[selected == 0] = -1
        pseudo_labels[rejected == 1] = -2

    # final pseudo labels
    num_src, num_total = get_num_classes(params.num_common, params.num_src_priv, params.num_tgt_priv)
    pseudo_labels[pseudo_labels == -2] = num_src
    assert (pseudo_labels != -1).all()

    # evaluation
    metrics = evaluate_T(params.num_common, params.num_src_priv, params.num_tgt_priv, lbls_T, cs_pseudo_labels,
                         rejected)
    return metrics
