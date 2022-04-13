from modules.logging.format_utils import format_measures


class MetricsLogger:
    def log(self, iteration, metrics_all, metrics_selected):
        pass

    def log_res(self, metrics_all):
        pass


class DefaultLogger(MetricsLogger):
    def log(self, iteration, metrics_all, metrics_selected):
        print('______')
        print(f'Iteration t={iteration}')
        print('all: ', format_measures(metrics_all))
        print('selected: ', format_measures(metrics_selected))

    def log_res(self, metrics_all):
        print('all: ', format_measures(metrics_all))
