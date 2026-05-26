# Magic numbers catalog — synoros-lib

Per the **no magic numbers** rule (`CLAUDE.md`, `README.md`), every numerical
constant in this library must be:

- **(a)** Derived from inputs (function of dims, structure, problem parameters)
- **(b)** A universal invariant (π, e, log N, 1/N, √N, mathematical identities)
- **(c)** Experimentally tuned with documented procedure + recorded scale-of-validity

This file catalogs every constant. Updating this file is a *precondition* for
adding any new constant.

**Constants don't transfer across scales.** A value that worked at one scale is
NOT assumed to work at another. Each entry should record the scale at which it
was determined and what to re-derive at scale.

---

## Status legend

- ✅ **Derived** — formula resolves to a value from inputs/invariants
- ⚖️ **Scale-invariant** — value is a ratio or dimensionless that auto-scales
- 🔬 **Experimentally set** — value tuned via documented sweep, scale-of-validity recorded
- ⚠️ **Unresolved** — currently picked, no derivation, marked for revisiting

---

## Library-wide ALLOWED literals (audit's safe-list, justified)

Same as synoros-substrate's parent library. See `src/synoros_lib/audit.py` for the
ALLOWED_LITERALS set:

| Literal | Justification |
|---|---|
| `0, 1, -1` (and float forms) | Mathematical identities |
| `1024, 1024.0` | Universal binary unit conversion (KB↔MB↔GB) |
| `1000, 1000.0` | Universal SI time conversion (s↔ms) AND RNG-stream-offset multiplier |
| `1e-9` (`numerical_floor_convention`) | Posited convention: anti-divide-by-zero floor. Chosen smaller than float32 eps (~1.19e-7) for safety across dtypes. Used pervasively in `1.0 / max(x, 1e-9)` and similar guards. |

---

## Cataloged constants (per-module)

(Currently empty — populate as the library grows.)

### manifolds

(no entries yet)

### topology

(no entries yet)

### algebra

(no entries yet)

### spectral

(no entries yet)

### info_geometry

(no entries yet)

### tensor_calculus

(no entries yet)

### optimization

(no entries yet)

### probability

(no entries yet)

---

## Re-derivation at scale

When a primitive in this library is used at a new scale (e.g., the consumer
goes from N=1000 to N=100000), the corresponding catalog entry should record
whether the constant transfers or needs re-derivation. Operations that scale
with N should be parameterized so the value auto-derives; ones that don't
need explicit re-validation at the new scale.
