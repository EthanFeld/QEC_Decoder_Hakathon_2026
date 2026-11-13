"""Validated adaptive CPU entry, with offline public-circuit asset bootstrap."""
import os as _os
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
_os.environ["QEC_DECODER_PROFILE"] = 'adaptive'
from setup_assets import ensure_assets as _ensure_assets
_boundary_bootstrap = _ensure_assets()
import organizer_decoder as _entry
from prior_model import DecoderConfig, DetectorModel, PriorTiltDecoder, model_from_dem, get_decoder_config
decode = _entry.decode
predict_proba = _entry.predict_proba
def get_startup_metadata():
    metadata = _entry.get_startup_metadata()
    metadata["boundary_bootstrap"] = _boundary_bootstrap
    return metadata
close = getattr(_entry, "close", lambda: None)
if hasattr(_entry, "get_decoder"):
    get_decoder = _entry.get_decoder
__all__ = ["decode", "predict_proba", "get_startup_metadata", "close", "DecoderConfig", "DetectorModel", "PriorTiltDecoder", "model_from_dem", "get_decoder_config"]
