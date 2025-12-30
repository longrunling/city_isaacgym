import argparse
import json
import os
import sys
import warnings

def build_scene(placements, use_materials):
    import trimesh
    from trimesh.transformations import rotation_matrix, translation_matrix

    cache = {}
    scene = trimesh.Scene()
    count = 0
    for rec in placements:
        src = rec.get('obj_src', rec.get('src'))
        pos = rec.get('position', [0.0, 0.0, 0.0])
        theta = rec.get('rotation', 0.0)
        if src and os.path.exists(src):
            resolved = src
        else:
            resolved = None
        if not resolved:
            print(f"Missing asset: {src}")
            continue
        try:
            if resolved in cache:
                base = cache[resolved]
            else:
                base = trimesh.load(resolved, force='mesh', skip_materials=(not use_materials))
                R_up = rotation_matrix(__import__('math').radians(90), [1, 0, 0])
                base.apply_transform(R_up)
                cache[resolved] = base
            minb, maxb = base.bounds
            center = (minb + maxb) / 2.0
            T_to_origin = translation_matrix(-center)
            Rz = rotation_matrix(theta, [0, 0, 1])
            tz = pos[2] + (-minb[2] + center[2])
            T_to_pos = translation_matrix([pos[0], pos[1], tz])
            transform = T_to_pos.dot(Rz).dot(T_to_origin)
            scene.add_geometry(base, node_name=f'obj_{count}', geom_name=f'geom::{resolved}', transform=transform)
            count += 1
        except Exception as e:
            print(f"Failed to add {resolved}: {e}")
    return scene, count

def copy_textures_and_rewrite_mtl(out_obj_path):
    try:
        base_dir = os.path.dirname(out_obj_path)
        name = os.path.splitext(os.path.basename(out_obj_path))[0]
        mtl_path = os.path.join(base_dir, f"{name}.mtl")
        if not os.path.exists(mtl_path):
            return
        textures_dir = os.path.join(base_dir, 'textures')
        os.makedirs(textures_dir, exist_ok=True)
        lines = []
        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip().startswith('map_Kd'):
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        tex_path = parts[1].strip()
                        src_path = tex_path
                        if not os.path.isabs(src_path):
                            # leave relative as is
                            pass
                        else:
                            if os.path.exists(src_path):
                                dst_path = os.path.join(textures_dir, os.path.basename(src_path))
                                try:
                                    import shutil
                                    if not os.path.exists(dst_path):
                                        shutil.copy2(src_path, dst_path)
                                    line = f"map_Kd textures/{os.path.basename(src_path)}\n"
                                except Exception:
                                    pass
                lines.append(line)
        with open(mtl_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Failed to process MTL textures: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate or visualize scene from placements.json and assets via src/obj_src')
    parser.add_argument('--placements', default=None, help='Path to placements.json; default: ./data/scene/placements.json')
    parser.add_argument('--out', default=None, help='Output OBJ path; default: ./data/scene/placed_scene.obj')
    parser.add_argument('--materials', action='store_true', help='Include materials/textures')
    parser.add_argument('--no-materials', action='store_true', help='Disable materials/textures')
    parser.add_argument('--copy-textures', action='store_true', help='Copy textures to output folder and rewrite MTL')
    parser.add_argument('--visualize', action='store_true', help='Visualize scene instead of exporting OBJ')
    args = parser.parse_args()

    placements_path = args.placements or os.path.join('.', 'data', 'scene', 'placements.json')
    out_obj = args.out or os.path.join('.', 'data', 'scene', 'placed_scene.obj')

    if not os.path.exists(placements_path):
        print(f"Placements JSON not found: {placements_path}")
        sys.exit(1)
    with open(placements_path, 'r', encoding='utf-8') as f:
        placements = json.load(f)

    use_materials = True
    if args.no_materials:
        use_materials = False
    elif args.materials:
        use_materials = True

    if args.visualize:
        try:
            scene, count = build_scene(placements, use_materials)
            print(f"Scene nodes: {count}")
            scene.show()
        except Exception as e:
            print(f"Failed to visualize: {e}")
            sys.exit(1)
    else:
        try:
            import trimesh
            scene, count = build_scene(placements, use_materials)
            os.makedirs(os.path.dirname(out_obj), exist_ok=True)
            scene.export(out_obj)
            print(f"Generated OBJ: {out_obj} (nodes={count})")
            if use_materials and args.copy_textures:
                copy_textures_and_rewrite_mtl(out_obj)
        except Exception as e:
            print(f"Failed to export: {e}")
            sys.exit(1)

if __name__ == '__main__':
    # silence PIL decompression bomb warnings optionally
    try:
        warnings.simplefilter('ignore', category=UserWarning)
    except Exception:
        pass
    main()
