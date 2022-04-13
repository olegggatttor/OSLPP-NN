def get_abbrev(source, target): return source[0].upper() + target[0].upper()


def format_measures(d): return ' '.join([f'{k}={v * 100:.2f}' for (k, v) in d.items()])
