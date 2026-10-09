"""incns -- Python job driver for the bessemer Stokes / Navier-Stokes solver.

Configure a case (parameters + analytic IC/BC/forcing) and run it, with no C++
recompile. All numerics stay in C++; Python exchanges only Parameters, field
callbacks, and scalar status. See CLAUDE.md "Python interface".
"""

from ._core import (
    Parameters,
    Case,
    BoxSpec,
    ScalingMode,
    Equation,
    StepControl,
    VelocityPreconditioner,
    GradDivScale,
    ConvectiveForm,
    OutflowCondition,
    ConvectionTreatment,
    RotationVelocityPC,
    SchurBlockType,
    SchurModel,
    PcQuadrature,
    BlockPCShape,
    APC,
    AmrThreshold,
    RotationSchurMode,
    RotationSchurCriterion,
    rank,
    size,
    on_root,
)

__all__ = [
    "Parameters",
    "Case",
    "BoxSpec",
    "ScalingMode",
    "Equation",
    "StepControl",
    "VelocityPreconditioner",
    "GradDivScale",
    "ConvectiveForm",
    "OutflowCondition",
    "ConvectionTreatment",
    "RotationVelocityPC",
    "SchurBlockType",
    "SchurModel",
    "PcQuadrature",
    "BlockPCShape",
    "APC",
    "AmrThreshold",
    "RotationSchurMode",
    "RotationSchurCriterion",
    "rank",
    "size",
    "on_root",
    "field",
]


class _CompiledField:
    """A numba-compiled analytic field: a raw C function pointer plus its dim.

    The case setters recognize it by its ``address``/``dim`` attributes and call
    the pointer directly in the C++ assembly loop -- no Python, no GIL.
    """

    __slots__ = ("address", "dim")

    def __init__(self, address, dim):
        self.address = address
        self.dim = dim


def field(dim):
    """Compile ``f(x, t, out)`` into the Tier-2 fast path.

    The decorated function is JIT-compiled with numba under the fixed C ABI
    ``void(const double* x, double t, double* out)``; write the ``dim``
    components of the field into ``out`` (``x[i]`` is the coordinate). Returns a
    compiled field the case setters accept directly.

    Example::

        @incns.field(dim=2)
        def u0(x, t, out):
            out[0] = x[1] * (1.0 - x[1])
            out[1] = 0.0
    """
    from numba import cfunc
    from numba.types import void, double, CPointer

    sig = void(CPointer(double), double, CPointer(double))

    def deco(pyfunc):
        compiled = cfunc(sig, nopython=True, cache=False)(pyfunc)
        return _CompiledField(compiled.address, dim)

    return deco
