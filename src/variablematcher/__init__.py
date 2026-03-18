"""
Variable Matcher Package.

A package for matching survey variables between SPSS SAV files using
embeddings and Google Cloud AI (Gemini) verification.

Examples
--------
>>> from variablematcher import Survey, VariableMatcher, GCPConfig
>>> config = GCPConfig(project_id='my-project')
>>> import pyreadstat
>>> df, meta = pyreadstat.read_sav('target.sav')
>>> target = Survey.from_sav(df, meta)
>>> df_cand, meta_cand = pyreadstat.read_sav('candidate.sav')
>>> candidate = Survey.from_sav(df_cand, meta_cand)
>>> matcher = VariableMatcher(gcp_config=config)
>>> result = matcher.fit(target, candidate).predict()
"""

__version__ = "0.1.0"

from .config import GCPConfig
from .encoder import EmbeddingEncoder
from .matcher import VariableMatcher
from .models import MatchResult, Value, ValueRecode, Variable, VariableMatch
from .survey import Survey

__all__ = [
    "__version__",
    "EmbeddingEncoder",
    "GCPConfig",
    "MatchResult",
    "Survey",
    "Value",
    "ValueRecode",
    "Variable",
    "VariableMatch",
    "VariableMatcher",
]
