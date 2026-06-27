# Simulation → Visualization Interface (Vinesh → Jasim)

**Status: AGREED.** This is the contract for what `solve_growth()` emits and
what `render_3d.render_volume()` consumes. Change it only by mutual agreement.

## One frame

| Property | Value |
|----------|-------|
| dtype | `np.float32` |
| shape | `(D, H, W)` = `(Z, Y, X)` — slices/depth first |
| semantics | **continuous density field**, values in `[0, 1]` |
| meaning | `1.0` = carrying capacity (packed tumor), `0.0` = no tumor |

It is **NOT** an integer label map. Jasim renders **by value** (isosurface /
volume opacity), e.g. an isosurface at `level=0.5`, with a continuous colormap
over `[0, 1]` (high density = denser tumor).

## A simulation run

`solve_growth(...)` returns `list[np.ndarray]` of length `timesteps + 1`
(index 0 is the initial frame). Every frame obeys the contract above and all
frames share one shape. Feed the whole list to Jasim for the animated view.

> Note: there is no necrotic/viable tissue labeling. The dataset has no
> necrotic ground truth, so we do not fabricate one. Render the density field
> by value only.

## Upstream (Philip/Chandan → Vinesh), for reference

`extract_volume()` must deliver the **same frame contract** (float32,
`(Z,Y,X)`, `[0,1]`) plus voxel `spacing` in mm via `params["spacing"]`.

## Example

```python
from tumor_pde_solver import solve_growth, dummy_volume

vol = dummy_volume()                      # or extract_volume(...) in Phase 3
frames = solve_growth(vol, timesteps=30, dt=0.1)

frames[0].dtype          # float32
frames[0].shape          # (D, H, W)
frames[0].min(), frames[0].max()   # within [0, 1]
```
