from typing import Dict
from .policy import (
    Net, Policy,
)
from .policy import (
    Seq2SeqModel
)

POLICY_CLASSES = {
    'Seq2SeqPolicy': Seq2SeqModel,
}

__all__ = [
    "Policy", "Net",
]