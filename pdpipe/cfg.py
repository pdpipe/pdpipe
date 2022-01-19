"""Configuration control for pdpipe."""

import birch


class CfgKey():
    LOAD_STAGE_ATTRIBUTES = 'LOAD_STAGE_ATTRIBUTES'


CFG = birch.Birch(
    namespace='pdpipe',
    defaults={
        CfgKey.LOAD_STAGE_ATTRIBUTES: 'True'
    },
    default_casters={
        CfgKey.LOAD_STAGE_ATTRIBUTES: birch.casters.true_false_caster,
    },
)

LOAD_STAGE_ATTRIBUTES = CFG[CfgKey.LOAD_STAGE_ATTRIBUTES]
