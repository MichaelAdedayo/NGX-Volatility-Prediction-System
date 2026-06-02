"""
NSE Volatility Prediction Package
B.Sc. Computer Science Final Year Project
"""

__version__ = "1.0.0"
__author__ = "Michael Adedayo Iseoluwa"

from .data_collection import NSEDataLoader, parse_volume
from .features import VolatilityFeatures, prepare_ml_dataset
from .garch_models import GARCHSuite, evaluate_garch_forecast
from .ml_models import MLVolatilitySuite, HybridModel
from .evaluation import StatisticalEvaluator

__all__ = [
    'NSEDataLoader',
    'VolatilityFeatures',
    'prepare_ml_dataset',
    'GARCHSuite',
    'evaluate_garch_forecast',
    'MLVolatilitySuite',
    'HybridModel',
    'StatisticalEvaluator'
]