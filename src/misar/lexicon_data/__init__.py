# -*- coding: utf-8 -*-
"""
Curated Multi-Dialect & Morphological Lexicon Data Packages for Misar.
"""

from typing import List
from ..lexicon import LexicalEntry

from .saudi_gulf import SAUDI_GULF_ENTRIES
from .egyptian import EGYPTIAN_ENTRIES
from .levantine import LEVANTINE_ENTRIES
from .maghrebi import MAGHREBI_ENTRIES
from .iraqi_yemeni import IRAQI_YEMENI_ENTRIES
from .msa_roots import MSA_ROOT_ENTRIES


def load_all_curated_entries() -> List[LexicalEntry]:
    """Aggregate and return all verified multi-dialect entries."""
    entries: List[LexicalEntry] = []
    entries.extend(SAUDI_GULF_ENTRIES)
    entries.extend(EGYPTIAN_ENTRIES)
    entries.extend(LEVANTINE_ENTRIES)
    entries.extend(MAGHREBI_ENTRIES)
    entries.extend(IRAQI_YEMENI_ENTRIES)
    entries.extend(MSA_ROOT_ENTRIES)
    return entries
