"""ML prediction model implementations."""
from __future__ import annotations

from quant_tools.predict.models.gradient_boost import GradientBoostModel
from quant_tools.predict.models.stats_forecast import StatsForecastModel
from quant_tools.predict.models.neural_forecast import NeuralForecastModel
from quant_tools.predict.models.prophet_model import ProphetModel
from quant_tools.predict.models.garch import GARCHModel
from quant_tools.predict.models.tft_model import TFTModel
from quant_tools.predict.models.patchtst_model import PatchTSTModel
from quant_tools.predict.models.nhits_model import NHiTSModel
from quant_tools.predict.models.timesnet_model import TimesNetModel
from quant_tools.predict.models.chronos_model import ChronosModel
from quant_tools.predict.models.timesfm_model import TimesFMModel
from quant_tools.predict.models.moirai_model import MoiraiModel

__all__ = [
    "GradientBoostModel",
    "StatsForecastModel",
    "NeuralForecastModel",
    "ProphetModel",
    "GARCHModel",
    "TFTModel",
    "PatchTSTModel",
    "NHiTSModel",
    "TimesNetModel",
    "ChronosModel",
    "TimesFMModel",
    "MoiraiModel",
]
