from mmdet.registry import DATA_SAMPLERS

# Keep the legacy symbol for backward compatibility in local imports.
SAMPLER = DATA_SAMPLERS


def build_sampler(cfg, default_args=None):
    return SAMPLER.build(cfg, default_args=default_args)
