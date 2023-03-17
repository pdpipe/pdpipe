from pdpipe.sklearn_stages import *
from pdpipe.col_generation import *
from pdpipe.basic_stages import *


STAGES_BY_CLASS = {
    'Bin': Bin,
    'Scale': Scale,
    'ColDrop': ColDrop
}
