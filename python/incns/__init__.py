"""incns -- Python job driver for the bessemer Stokes / Navier-Stokes solver.

Configure a case (parameters + analytic IC/BC/forcing) and run it, with no C++
recompile. All numerics stay in C++; Python exchanges only Parameters, field
callbacks, scalar status and the run summary. See docs/developing.md 5.7.

Three ways in:

* ``incns.run_deck(deck)`` runs a deck end to end exactly as ``run_case``
  does (flows and solids) and returns its summary as a dict.
* ``incns.Parameters.from_dict({...})`` / ``from_yaml(path)`` /
  ``from_yaml_string(text)`` reach every deck key (``incns.deck_keys()``
  lists them); the common ones are also attributes of ``Parameters``.
* ``incns.Case(params)`` steps a flow from Python, with analytic fields
  (``@incns.field`` or plain callables) and scalar results.
"""

import json as _json

from ._core import (
    Parameters,
    Case,
    BoxSpec,
    ScalingMode,
    Equation,
    StepControl,
    SolverFailure,
    VelocityPreconditioner,
    GradDivScale,
    ConvectiveForm,
    OutflowCondition,
    ConvectionTreatment,
    VelocityMass,
    HangingMass,
    VectorDiffusionStorage,
    MeshGeometry,
    SubgridModel,
    SubgridOptions,
    OuterMethod,
    InnerTolerance,
    SchurSolveOptions,
    ToleranceReference,
    InitialGuess,
    ProfileOutput,
    RotationVelocityPC,
    SchurBlockType,
    SchurModel,
    PcQuadrature,
    BlockPCShape,
    APC,
    APcgPC,
    AmrThreshold,
    RotationSchurMode,
    RotationSchurCriterion,
    deck_keys,
    rank,
    size,
    on_root,
)
from ._core import _run, _read_summary

__all__ = [
    "Parameters",
    "Case",
    "BoxSpec",
    "ScalingMode",
    "Equation",
    "StepControl",
    "SolverFailure",
    "VelocityPreconditioner",
    "GradDivScale",
    "ConvectiveForm",
    "OutflowCondition",
    "ConvectionTreatment",
    "VelocityMass",
    "HangingMass",
    "VectorDiffusionStorage",
    "MeshGeometry",
    "SubgridModel",
    "SubgridOptions",
    "OuterMethod",
    "InnerTolerance",
    "SchurSolveOptions",
    "ToleranceReference",
    "InitialGuess",
    "ProfileOutput",
    "RotationVelocityPC",
    "SchurBlockType",
    "SchurModel",
    "PcQuadrature",
    "BlockPCShape",
    "APC",
    "APcgPC",
    "AmrThreshold",
    "RotationSchurMode",
    "RotationSchurCriterion",
    "deck_keys",
    "run_deck",
    "rank",
    "size",
    "on_root",
    "field",
]


def _from_dict(deck):
    """Parameters from a deck given as a dict (nested like the YAML).

    JSON is YAML, so the dict goes through the same loader as a deck file:
    every key is reachable, defaults apply to what is left out, and an
    unknown key raises ValueError with the loader's suggestion.
    """
    return Parameters.from_yaml_string(_json.dumps(deck))


Parameters.from_dict = staticmethod(_from_dict)


def run_deck(deck, summary=""):
    """Run a deck end to end, as ``run_case`` does, and return its summary.

    ``deck`` is a path to a YAML deck, a dict, or a ``Parameters``. A flow
    starts from the deck's named initial condition with its boundary groups
    (every real boundary must be covered); ``equation: linear_elasticity``
    runs the solid. The summary is written to ``summary`` (default
    ``<output.path>/<output.name>_summary.json``) at the start and the end, and
    returned as a dict on every rank; its ``status`` is ``ok`` or
    ``diverged``. SPMD: call it on every rank.
    """
    if isinstance(deck, dict):
        params = _from_dict(deck)
    elif isinstance(deck, Parameters):
        params = deck
    else:
        params = Parameters.from_yaml(str(deck))
    path, _ok = _run(params, summary)
    return _read_summary(path)


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
