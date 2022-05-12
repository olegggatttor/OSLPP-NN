import math
from functools import reduce

from torch import nn

from modules.scoring.metrics import *
from modules.scoring.metrics_ensemble import get_mean_confs, get_mean_margins, get_score_total_U, \
    get_weighted_scores_multi, get_all_tops_multi
from modules.types.selection_types import SelectRejectMode


def select_initial_rejected(all_predictions, n_r, mode: SelectRejectMode, weights, tops, aug_preds):
    rej_shape = len(all_predictions[0].max(dim=1)[0])
    is_rejected = np.zeros((rej_shape,), dtype=np.int)
    if mode == SelectRejectMode.CONFIDENCE:
        assert all_predictions.shape[0] == 1
        pseudo_probs = all_predictions[0].max(dim=1)[0].numpy()
        scores = get_confidence_scores(pseudo_probs)
    elif mode == SelectRejectMode.ENTROPY:
        assert all_predictions.shape[0] == 1
        scores = get_entropy_scores(all_predictions[0])
    elif mode == SelectRejectMode.MARGIN:
        assert all_predictions.shape[0] == 1
        scores = get_margin_scores(all_predictions[0])
    elif mode == SelectRejectMode.AUGMENTATION_SIMILARITY:
        assert all_predictions.shape[0] == 1
        assert aug_preds is not None

        preds = all_predictions[0]

        ce_loss = nn.CrossEntropyLoss(reduction='none')

        scores = (ce_loss(aug_preds, preds).detach().cpu().numpy() +
                  ce_loss(preds, aug_preds).detach().cpu().numpy()) / 2
    elif mode == SelectRejectMode.WEIGHTED:
        assert all_predictions.shape[0] == 1
        assert weights is not None

        pseudo_probs = all_predictions[0].max(dim=1)[0].numpy()
        scores = get_weighted_scores(pseudo_probs, all_predictions[0], weights)
    elif mode == SelectRejectMode.TOPS:
        assert all_predictions.shape[0] == 1
        assert tops is not None

        pseudo_probs = all_predictions[0].max(dim=1)[0].numpy()
        all_tops = get_all_tops(pseudo_probs, all_predictions[0], tops)
        rejects_tops = list(map(lambda scores: np.argsort(scores)[:2 * n_r], all_tops))
        is_rejected[reduce(np.intersect1d, rejects_tops)] = 1
        return is_rejected
    elif mode == SelectRejectMode.CONFIDENCE_MULTI:
        scores = get_mean_confs(all_predictions)
    elif mode == SelectRejectMode.MARGIN_MULTI:
        scores = get_mean_margins(all_predictions)
    elif mode == SelectRejectMode.TOTAL_U:
        scores = get_score_total_U(all_predictions)
    elif mode == SelectRejectMode.WEIGHTED_MULTI:
        assert weights is not None

        scores = get_weighted_scores_multi(all_predictions, weights)
    elif mode == SelectRejectMode.TOPS_MULTI:
        assert tops is not None

        all_tops = get_all_tops_multi(all_predictions, tops)
        rejects_tops = list(map(lambda scores: np.argsort(scores)[:2 * n_r], all_tops))
        if len(reduce(np.intersect1d, rejects_tops)) == 0:
            print('[INIT_REJECTED]: WEIGHTED WAS USED INSTEAD OF TOPS')
            scores = get_weighted_scores_multi(all_predictions, np.array([0.33, 0.33, 0.33]))
            is_rejected[np.argsort(scores)[:n_r]] = 1
            return is_rejected
        is_rejected[reduce(np.intersect1d, rejects_tops)] = 1
        return is_rejected
    else:
        raise Exception("Wrong select/reject mode.")
    is_rejected[np.argsort(scores)[:n_r]] = 1
    return is_rejected


def get_new_rejected(all_predictions, selected, rejected, mode: SelectRejectMode, weights, tops, aug_preds):
    def reject_by_scores(scores, rejected, selected):
        return scores < min(scores[rejected == 1].max(), scores[selected == 1].min())

    if mode == SelectRejectMode.CONFIDENCE:
        assert all_predictions.shape[0] == 1

        pseudo_probs = all_predictions[0].max(dim=1)[0].numpy()
        scores = get_confidence_scores(pseudo_probs)
    elif mode == SelectRejectMode.ENTROPY:
        assert all_predictions.shape[0] == 1

        scores = get_entropy_scores(all_predictions[0])
    elif mode == SelectRejectMode.MARGIN:
        assert all_predictions.shape[0] == 1

        scores = get_margin_scores(all_predictions[0])
    elif mode == SelectRejectMode.AUGMENTATION_SIMILARITY:
        assert all_predictions.shape[0] == 1
        assert aug_preds is not None

        preds = all_predictions[0]

        ce_loss = nn.CrossEntropyLoss(reduction='none')

        scores = (ce_loss(aug_preds, preds).detach().cpu().numpy() +
                  ce_loss(preds, aug_preds).detach().cpu().numpy()) / 2
    elif mode == SelectRejectMode.WEIGHTED:
        assert all_predictions.shape[0] == 1
        assert weights is not None

        pseudo_probs = all_predictions[0].max(dim=1)[0].numpy()
        scores = get_weighted_scores(pseudo_probs, all_predictions[0], weights)
    elif mode == SelectRejectMode.TOPS:

        assert all_predictions.shape[0] == 1
        assert tops is not None
        pseudo_probs = all_predictions[0].max(dim=1)[0].numpy()
        all_rejects = get_all_tops(pseudo_probs, all_predictions[0], tops)

        mask = np.logical_and.reduce(list(map(lambda x: reject_by_scores(x, rejected, selected), all_rejects)))
        return mask
    elif mode == SelectRejectMode.CONFIDENCE_MULTI:
        scores = get_mean_confs(all_predictions)
    elif mode == SelectRejectMode.MARGIN_MULTI:
        scores = get_mean_margins(all_predictions)
    elif mode == SelectRejectMode.TOTAL_U:
        scores = get_score_total_U(all_predictions)
    elif mode == SelectRejectMode.WEIGHTED_MULTI:
        assert weights is not None

        scores = get_weighted_scores_multi(all_predictions, weights)
    elif mode == SelectRejectMode.TOPS_MULTI:
        assert tops is not None

        all_rejects = get_all_tops_multi(all_predictions, tops)

        mask = np.logical_and.reduce(list(map(lambda x: reject_by_scores(x, rejected, selected), all_rejects)))
        return mask
    else:
        raise Exception("Wrong select/reject mode.")
    scores = torch.tensor(scores)
    return (scores < min(scores[rejected == 1].max().item(), scores[selected == 1].min().item())).int()


def select_closed_set_pseudo_labels_by_mode(class_indices, selected, all_predictions,
                                            t, T, Nc,
                                            mode: SelectRejectMode,
                                            uniform_ratio, balanced, weights, tops, aug_preds):
    def get_threshold(all_scores, Nc, top_length_multiplier=2):
        idx_tr = math.floor(Nc * (1 - t / (T - 1)))
        return all_scores[max(len(all_scores) - (len(all_scores) - idx_tr) * top_length_multiplier, 0)]

    def get_mask(arr, thresh):
        return arr >= thresh

    thresh_pos = math.floor(Nc * (1 - t / (T - 1)))
    if mode == SelectRejectMode.CONFIDENCE:
        assert all_predictions.shape[0] == 1
        predictions = all_predictions[0]
        pseudo_probs = predictions.max(dim=1)[0].numpy()

        scores = pseudo_probs[class_indices]
        class_probs = np.sort(scores)
        if balanced and t != T - 1:
            to_draw_border = math.floor(uniform_ratio * t / (T - 1))
            threshold = list(reversed(class_probs))[min(to_draw_border, Nc - 1)]
        else:
            threshold = class_probs[thresh_pos]
    elif mode == SelectRejectMode.ENTROPY:
        assert all_predictions.shape[0] == 1
        predictions = all_predictions[0]
        scores = get_entropy_scores(predictions[class_indices])
        entropy = np.sort(scores)
        threshold = entropy[thresh_pos]
    elif mode == SelectRejectMode.MARGIN:
        assert all_predictions.shape[0] == 1
        predictions = all_predictions[0]
        scores = get_margin_scores(predictions[class_indices])
        margin = np.sort(scores)
        threshold = margin[thresh_pos]
    elif mode == SelectRejectMode.AUGMENTATION_SIMILARITY:
        assert all_predictions.shape[0] == 1
        assert aug_preds is not None

        preds = all_predictions[0][class_indices]
        aug_preds_indexed = aug_preds[class_indices]
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        scores = (ce_loss(aug_preds_indexed, preds).detach().cpu().numpy() +
                  ce_loss(preds, aug_preds_indexed).detach().cpu().numpy()) / 2

        similarity = np.sort(scores)
        threshold = similarity[thresh_pos]
    elif mode == SelectRejectMode.WEIGHTED:
        assert all_predictions.shape[0] == 1
        assert weights is not None

        predictions = all_predictions[0]

        pseudo_probs = predictions.max(dim=1)[0].numpy()

        class_probs = pseudo_probs[class_indices]
        class_predictions = predictions[class_indices]
        scores = get_weighted_scores(class_probs, class_predictions, weights)
        weighted = np.sort(scores)
        threshold = weighted[thresh_pos]
    elif mode == SelectRejectMode.TOPS:
        assert all_predictions.shape[0] == 1
        assert tops is not None

        predictions = all_predictions[0]
        pseudo_probs = predictions.max(dim=1)[0].numpy()

        class_probs = pseudo_probs[class_indices]
        class_predictions = predictions[class_indices]

        all_tops = get_all_tops(class_probs, class_predictions, tops)

        sorted_all_tops = list(map(np.sort, all_tops))
        threshold_all_tops = list(map(lambda x: get_threshold(x, Nc), sorted_all_tops))
        masks_all_tops = list(map(lambda x: get_mask(*x), zip(all_tops, threshold_all_tops)))

        selected_indices = class_indices[np.logical_and.reduce(masks_all_tops)]
        assert (selected[selected_indices] == 0).all()
        selected[selected_indices] = 1
        return selected
    elif mode == SelectRejectMode.CONFIDENCE_MULTI:
        class_preds = all_predictions[:, class_indices, :]
        scores = get_mean_confs(class_preds)
        class_probs = np.sort(scores)
        threshold = class_probs[thresh_pos]
    elif mode == SelectRejectMode.MARGIN_MULTI:
        class_preds = all_predictions[:, class_indices, :]
        scores = get_mean_margins(class_preds)
        margin = np.sort(scores)
        threshold = margin[thresh_pos]
    elif mode == SelectRejectMode.TOTAL_U:
        class_preds = all_predictions[:, class_indices, :]
        scores = get_score_total_U(class_preds)
        total_u = np.sort(scores)
        threshold = total_u[thresh_pos]
    elif mode == SelectRejectMode.WEIGHTED_MULTI:
        assert weights is not None

        scores = get_weighted_scores_multi(all_predictions[:, class_indices, :], weights)
        weighted = np.sort(scores)
        threshold = weighted[thresh_pos]
    elif mode == SelectRejectMode.TOPS_MULTI:
        class_predictions = all_predictions[:, class_indices, :]

        all_tops = get_all_tops_multi(class_predictions, tops)

        sorted_all_tops = list(map(np.sort, all_tops))
        threshold_all_tops = list(map(lambda x: get_threshold(x, Nc), sorted_all_tops))
        masks_all_tops = list(map(lambda x: get_mask(*x), zip(all_tops, threshold_all_tops)))

        if len(np.logical_and.reduce(masks_all_tops)) == 0:
            print('[GET_SELECTED]: WEIGHTED WAS USED INSTEAD OF TOPS')
            scores = get_weighted_scores_multi(all_predictions[:, class_indices, :], np.array([0.33, 0.33, 0.33]))
            weighted = np.sort(scores)
            threshold = weighted[thresh_pos]
        else:
            selected_indices = class_indices[np.logical_and.reduce(masks_all_tops)]
            assert (selected[selected_indices] == 0).all()
            selected[selected_indices] = 1
            return selected
    else:
        raise Exception("Wrong select/reject mode.")
    mask = scores >= threshold
    selected_indices = class_indices[mask]
    assert (selected[selected_indices] == 0).all()
    selected[selected_indices] = 1
    return selected
