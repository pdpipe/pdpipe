"""Configuration control for pdpipe."""

import birch


class CfgKey():
    LOAD_STAGE_ATTRIBUTES = 'LOAD_STAGE_ATTRIBUTES'
    LOAD_CORE_AS_MODULE = 'LOAD_CORE_AS_MODULE'


CFG = birch.Birch(
    namespace='pdpipe',
    defaults={
        CfgKey.LOAD_STAGE_ATTRIBUTES: 'True',
        CfgKey.LOAD_CORE_AS_MODULE: 'False',
    },
    default_casters={
        CfgKey.LOAD_STAGE_ATTRIBUTES: birch.casters.true_false_caster,
        CfgKey.LOAD_CORE_AS_MODULE: birch.casters.true_false_caster,
    },
)

LOAD_STAGE_ATTRIBUTES = CFG[CfgKey.LOAD_STAGE_ATTRIBUTES]
LOAD_CORE_AS_MODULE = CFG[CfgKey.LOAD_CORE_AS_MODULE]
