## Purpose
Create a standalone Python script that builds a full scene from `placements.json` in `data/scene` and OBJ assets in `data/building/obj`, `data/car/obj`, and `data/ped/obj`. It should work for users who only have the `data` folder.

## Inputs
- `--data-dir`: Path to the `data` directory (default: `./data`).
- `--placements`: Path to placements JSON (default: `<data-dir>/scene/placements.json`).
- `--out`: Output OBJ path (default: `<data-dir>/scene/placed_scene.obj`).
- `--materials`: Include materials/textures (default: `true`).
- `--copy-textures`: Copy referenced textures into `<out>` directory and rewrite MTL to relative paths (default: `false`).
- `--visualize`: Open an interactive viewer instead of exporting OBJ (default: `false`).

## Path Resolution
- The JSON `src` may contain absolute paths from a different machine. The script will:
  - Extract the filename from each `src`.
  - Search for the file under `data/building/obj`, `data/car/obj`, and `data/ped/obj`.
  - Use the first match found; error if not found.

## Core Logic
- Use `trimesh` to load each unique OBJ (once), rotate upright (+90° around X), cache it by filename.
- For each placement record:
  - Compute per-instance transform: translate to origin by mesh center, rotate around Z (`rotation`), then translate to `[x, y, z]` adjusted to place base on `z=0`.
  - Add the shared geometry to a `trimesh.Scene` with the per-instance transform (no per-instance mesh copies).

## Materials & Textures
- If `--materials` is true:
  - Load with `skip_materials=False` so textures/MTL are preserved.
  - When exporting OBJ, `trimesh` will emit an MTL and reference texture files.
- If `--copy-textures` is true:
  - Parse emitted MTL, copy referenced textures into the output directory (or `textures/`), and rewrite MTL references to relative paths.

## Output Modes
- `--visualize`: Build the `Scene` and call `scene.show()` (no OBJ written).
- Default: Export combined OBJ/MTL at `--out` using the shared-geometry scene for memory efficiency.

## CLI & Script Location
- Add `scripts/generate_scene.py` at repo root.
- Dependencies: `trimesh` (and `pyglet` only for `--visualize`).

## Error Handling
- Validate inputs; clear messages if placements or asset directories are missing.
- Report any records whose asset filename cannot be found.
- Continue past missing assets, and summarize counts at the end.

## Verification
- Run against existing `data` to confirm OBJ export and viewer modes.
- Spot-check a few placements to verify transforms and ground alignment.

## Next Steps
- Implement the script and wire up the CLI.
- Optionally add README usage snippet: `python scripts/generate_scene.py --data-dir ./data --materials --out ./data/scene/placed_scene.obj`. 