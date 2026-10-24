import numpy as np
from lib import misc


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of algorithm-specific hyperparameters.
    Each entry is a (default, random) tuple.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    if algorithm == 'LWS':
        _hparam('lw_weight', 1, lambda r: r.choice([1, 2]))
    elif algorithm == 'DIRK':
        _hparam('momentum', 0.99, lambda r: r.choice([0.5, 0.9, 0.99]))
        _hparam('moco_queue', 8192, lambda r: r.choice([4096, 8192]))
        _hparam('feat_dim', 128, lambda r: r.choice([64, 128, 256]))
        _hparam('prot_start', 1, lambda r: r.choice([1]))
        _hparam('weight', 1.0, lambda r: r.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
        _hparam('dist_temperature', 0.4, lambda r: r.choice([0.2, 0.4, 0.6]))
        _hparam('feat_temperature', 0.07, lambda r: r.choice([0.03, 0.05, 0.07, 0.09]))
    elif algorithm == 'ABLE':
        _hparam('feat_dim', 128, lambda r: r.choice([64, 128, 256]))
        _hparam('loss_weight', 1.0, lambda r: r.choice([0.5, 1.0, 2.0]))
        _hparam('temperature', 0.07, lambda r: r.choice([0.03, 0.05, 0.07, 0.09]))
    elif algorithm == 'PiCO':
        _hparam('prot_start', 1, lambda r: r.choice([1, 100]))
        _hparam('momentum', 0.9, lambda r: r.choice([0.8, 0.9]))
        _hparam('feat_dim', 128, lambda r: r.choice([64, 128, 256]))
        _hparam('moco_queue', 8192, lambda r: r.choice([4096, 8192]))
        _hparam('moco_m', 0.999, lambda r: r.choice([0.9, 0.999]))
        _hparam('proto_m', 0.99, lambda r: r.choice([0.99, 0.9]))
        _hparam('loss_weight', 0.5, lambda r: r.choice([0.5, 1.0]))
        _hparam('conf_ema_range', '0.95,0.8', lambda r: r.choice(['0.95,0.8']))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
