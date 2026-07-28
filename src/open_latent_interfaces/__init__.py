"""Tools for discovering and intervening on native latent interfaces."""

from open_latent_interfaces.activations import ActivationCapture, CapturedLayer
from open_latent_interfaces.arithmetic_coordinates import (
    ArithmeticCoordinateManifest,
    TokenLocalTransportWriter,
)
from open_latent_interfaces.interpretability import (
    InterpretabilityArtifact,
    LatentSite,
    MethodProvenance,
    corroborate,
)
from open_latent_interfaces.native_coordinates import (
    NativeCoordinateManifest,
    NativeCoordinateWriter,
    fit_digit_prototypes,
)
from open_latent_interfaces.probes import (
    BinaryRidgeProbe,
    CategoricalRidgeProbe,
    ScalarRidgeProbe,
)

__all__ = [
    "ActivationCapture",
    "ArithmeticCoordinateManifest",
    "BinaryRidgeProbe",
    "CategoricalRidgeProbe",
    "CapturedLayer",
    "InterpretabilityArtifact",
    "LatentSite",
    "MethodProvenance",
    "NativeCoordinateManifest",
    "NativeCoordinateWriter",
    "ScalarRidgeProbe",
    "TokenLocalTransportWriter",
    "corroborate",
    "fit_digit_prototypes",
]

__version__ = "0.1.0"
