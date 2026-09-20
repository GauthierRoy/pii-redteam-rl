"""Generation-backend contract (M01.3).

Records precision/decoder options with a no-speculation reference path and raises
useful errors for unsupported combinations. Real backends (FP8, speculative, DFlash)
arrive in S01-S04; the fake CPU backend used by the smoke test supports fp32 only.
"""

from __future__ import annotations

from dataclasses import dataclass

REFERENCE_PRECISION = "fp32"
REFERENCE_DECODER = "reference"

# Known names (helpful errors); only the reference pair runs at M01.
KNOWN_PRECISIONS = ("fp32", "fp16", "bf16", "fp8")
KNOWN_DECODERS = ("reference", "speculative-ngram", "dflash")


class UnsupportedBackendError(ValueError):
    """Raised when a precision/decoder combination cannot run here."""


@dataclass(frozen=True)
class BackendSpec:
    precision: str = REFERENCE_PRECISION
    decoder: str = REFERENCE_DECODER


def validate_backend(spec: dict | BackendSpec) -> BackendSpec:
    """Normalize a backend spec dict, raising UnsupportedBackendError with a useful message."""
    precision = spec["precision"] if isinstance(spec, dict) else spec.precision
    decoder = spec["decoder"] if isinstance(spec, dict) else spec.decoder
    if precision not in KNOWN_PRECISIONS:
        raise UnsupportedBackendError(
            f"unknown precision {precision!r}; known: {', '.join(KNOWN_PRECISIONS)}"
        )
    if decoder not in KNOWN_DECODERS:
        raise UnsupportedBackendError(
            f"unknown decoder {decoder!r}; known: {', '.join(KNOWN_DECODERS)}"
        )
    if precision != REFERENCE_PRECISION:
        raise UnsupportedBackendError(
            f"precision {precision!r} is not served by the M01 fake CPU backend "
            f"(supports {REFERENCE_PRECISION!r} only); "
            "FP8 validation is S01 work on real hardware"
        )
    if decoder != REFERENCE_DECODER:
        track = "S03" if "speculative" in decoder else "S04"
        raise UnsupportedBackendError(
            f"decoder {decoder!r} requires the {track} validated path; "
            f"M01 smoke runs {REFERENCE_DECODER!r} only"
        )
    return BackendSpec(precision=precision, decoder=decoder)


def capabilities() -> dict:
    """Feature detection: what the current (M01 fake) backend supports."""
    return {
        "precisions": [REFERENCE_PRECISION],
        "decoders": [REFERENCE_DECODER],
        "speculative": False,
        "quantized": False,
        "note": "M01 fake CPU backend; S00-S05 add real capabilities with tests",
    }
