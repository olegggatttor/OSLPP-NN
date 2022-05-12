import pandas as pd

from modules.logging.format_utils import format_measures


class MetricsLogger:
    def log(self, iteration, metrics_all, metrics_selected):
        pass

    def log_res(self, metrics_all):
        pass

    def augmentation_save(self, source, target, seed, t, transform, similarity_common, similarity_private,
                          aug_common_conf, aug_common_inv_entropy, aug_common_margin,
                          aug_private_conf, aug_private_inv_entropy, aug_private_margin,
                          no_aug_common_conf, no_aug_common_inv_entropy, no_aug_common_margin,
                          no_aug_private_conf, no_aug_private_inv_entropy, no_aug_private_margin
                          ):
        pass


class DefaultLogger(MetricsLogger):
    def __init__(self):
        self.df = pd.DataFrame({'source': [], 'target': [], 'seed': [], 't': [], 'transform': []
                                   , 'similarity_common': [], 'similarity_private': []
                                   , 'aug_common_conf': [], 'aug_common_inv_entropy': [], 'aug_common_margin': []
                                   , 'aug_private_conf': [], 'aug_private_inv_entropy': [], 'aug_private_margin': []
                                   , 'no_aug_common_conf': [], 'no_aug_common_inv_entropy': [], 'no_aug_common_margin': []
                                   , 'no_aug_private_conf': [], 'no_aug_private_inv_entropy': [], 'no_aug_private_margin': []
                                })

    def log(self, iteration, metrics_all, metrics_selected):
        print('______')
        print(f'Iteration t={iteration}')
        print('all: ', format_measures(metrics_all))
        print('selected: ', format_measures(metrics_selected))

    def log_res(self, metrics_all):
        print('all: ', format_measures(metrics_all))

    def augmentation_save(self, source, target, seed, t, transform, similarity_common, similarity_private,
                          aug_common_conf, aug_common_inv_entropy, aug_common_margin,
                          aug_private_conf, aug_private_inv_entropy, aug_private_margin,
                          no_aug_common_conf, no_aug_common_inv_entropy, no_aug_common_margin,
                          no_aug_private_conf, no_aug_private_inv_entropy, no_aug_private_margin
                          ):
        self.df = self.df.append({'source': source, 'target': target, 'seed': seed, 't': t, 'transform': transform
                                     , 'similarity_common': similarity_common, 'similarity_private': similarity_private
                                     , 'aug_common_conf': aug_common_conf,
                                  'aug_common_inv_entropy': aug_common_inv_entropy,
                                  'aug_common_margin': aug_common_margin
                                     , 'aug_private_conf': aug_private_conf,
                                  'aug_private_inv_entropy': aug_private_inv_entropy,
                                  'aug_private_margin': aug_private_margin

                                     , 'no_aug_common_conf': no_aug_common_conf,
                                  'no_aug_common_inv_entropy': no_aug_common_inv_entropy,
                                  'no_aug_common_margin': no_aug_common_margin
                                     , 'no_aug_private_conf': no_aug_private_conf,
                                  'no_aug_private_inv_entropy': no_aug_private_inv_entropy,
                                  'no_aug_private_margin': no_aug_private_margin
                                  }, ignore_index=True)
