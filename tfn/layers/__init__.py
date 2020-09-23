from tensorflow.keras.utils import get_custom_objects

from tfn.layers.utils import shifted_softplus, tfn_mae

from .atomic_images import (
    OneHot,
    DistanceMatrix,
    KernelBasis,
    GaussianBasis,
    AtomicNumberBasis,
    Unstandardization,
    DummyAtomMasking,
    CutoffLayer,
    CosineCutoff,
    TanhCutoff,
    LongTanhCutoff,
)

from .utility_layers import Preprocessing, UnitVectors, MaskedDistanceMatrix
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
        "shifted_softplus": shifted_softplus,
        "tfn_mae": tfn_mae,
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
        MaskedDistanceMatrix.__name__: MaskedDistanceMatrix,
        OneHot.__name__: OneHot,
        DistanceMatrix.__name__: DistanceMatrix,
        KernelBasis.__name__: KernelBasis,
        GaussianBasis.__name__: GaussianBasis,
        AtomicNumberBasis.__name__: AtomicNumberBasis,
        Unstandardization.__name__: Unstandardization,
        DummyAtomMasking.__name__: DummyAtomMasking,
        CutoffLayer.__name__: CutoffLayer,
        CosineCutoff.__name__: CosineCutoff,
        TanhCutoff.__name__: TanhCutoff,
        LongTanhCutoff.__name__: LongTanhCutoff,
    }
)
