from tensorflow.keras.utils import get_custom_objects

from ..utils import shifted_softplus
from .utility_layers import Preprocessing, UnitVectors
from .radial_factories import RadialFactory, DenseRadialFactory, Radial
from .layers import (
    EquivariantLayer,
    Convolution,
    HarmonicFilter,
    SelfInteraction,
    EquivariantActivation,
)
from .molecular_layers import (
    MolecularConvolution,
    MolecularSelfInteraction,
    MolecularActivation,
)


get_custom_objects().update(
    {
        "ssp": shifted_softplus,
        RadialFactory.__name__: RadialFactory,
        DenseRadialFactory.__name__: DenseRadialFactory,
        Radial.__name__: Radial,
        EquivariantLayer.__name__: EquivariantLayer,
        Convolution.__name__: Convolution,
        MolecularConvolution.__name__: MolecularConvolution,
        HarmonicFilter.__name__: HarmonicFilter,
        SelfInteraction.__name__: SelfInteraction,
        MolecularSelfInteraction.__name__: MolecularSelfInteraction,
        EquivariantActivation.__name__: EquivariantActivation,
        MolecularActivation.__name__: MolecularActivation,
        Preprocessing.__name__: Preprocessing,
        UnitVectors.__name__: UnitVectors,
    }
)
