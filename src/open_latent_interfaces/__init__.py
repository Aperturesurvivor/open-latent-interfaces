"""Tools for discovering and intervening on native latent interfaces."""

from open_latent_interfaces.activations import (
    ActivationCapture,
    CapturedLayer,
    CapturedTokenPositions,
)
from open_latent_interfaces.arithmetic_coordinates import (
    ArithmeticCoordinateManifest,
    TokenLocalTransportWriter,
)
from open_latent_interfaces.hybrid_graft import HybridGraftManifest
from open_latent_interfaces.interpretability import (
    InterpretabilityArtifact,
    LatentSite,
    MethodProvenance,
    corroborate,
)
from open_latent_interfaces.model_onboarding import (
    ModelOnboardingSpec,
    candidate_hidden_state_indices,
)
from open_latent_interfaces.native_coordinates import (
    NativeCoordinateManifest,
    NativeCoordinateWriter,
    fit_digit_prototypes,
)
from open_latent_interfaces.operand_reader import (
    NearestCentroidDigitReader,
    OperandReaderManifest,
    OperandTokenPositions,
    fit_nearest_centroid_digit_reader,
    locate_operand_digit_tokens,
    reconstruct_decimal_digits,
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
    "CapturedTokenPositions",
    "InterpretabilityArtifact",
    "HybridGraftManifest",
    "LatentSite",
    "MethodProvenance",
    "ModelOnboardingSpec",
    "NativeCoordinateManifest",
    "NativeCoordinateWriter",
    "NearestCentroidDigitReader",
    "OperandReaderManifest",
    "OperandTokenPositions",
    "ScalarRidgeProbe",
    "TokenLocalTransportWriter",
    "corroborate",
    "candidate_hidden_state_indices",
    "fit_digit_prototypes",
    "fit_nearest_centroid_digit_reader",
    "locate_operand_digit_tokens",
    "reconstruct_decimal_digits",
]

__version__ = "0.1.0"
