#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import random
try:
    from .vertex import Vertex
    from .growth import organic
    from .check import SimpleGlobal, check, rebuild_tree
except Exception:
    # Allow running this file directly (python main.py)
    from vertex import Vertex
    from growth import organic
    from check import SimpleGlobal, check, rebuild_tree
import os
try:
    import trimesh
except Exception:
    trimesh = None
try:
    from .placement import place_buildings, export_scene, place_cars, place_peds
except Exception:
    from placement import place_buildings, export_scene, place_cars, place_peds
try:
    from .placement import export_placement_json, generate_obj_from_placement, visualize_placement_scene
except Exception:
    from placement import export_placement_json, generate_obj_from_placement, visualize_placement_scene


def run(params):
    # create global
    # seed control: if provided, make all random draws deterministic
    seed = params.get('seed', None)
    if seed is not None:
        try:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            print(f"[simple_citygen] using seed={seed}")
        except Exception:
            print(f"[simple_citygen] invalid seed value: {params.get('seed')}")

    def generate_road(params, seed=None, visualize=False, out_image=None):
        """Generate a roadmap graph using current params.

        If seed is not None, reseed python and numpy RNGs to make generation deterministic per-seed.
        If visualize is True and out_image is provided, save a simple roadmap visualization (pre-scale) to out_image.
        Returns the generated SimpleGlobal instance (with coords scaled according to params['scale']).
        """
        # optional reseed for deterministic variants
        if seed is not None:
            try:
                s = int(seed)
                random.seed(s)
                np.random.seed(s)
            except Exception:
                pass

        g_local = SimpleGlobal()
        axiom_coords = params.get("axiom", [[2,0],[3,0],[0,-2],[0,-3],[0,2],[0,3],[-2,0],[-3,0]])
        axiom = [Vertex(c) for c in axiom_coords]
        for v in axiom:
            other = min([u for u in axiom if u is not v], key=lambda x: np.linalg.norm(x.coords - v.coords))
            v.neighbours = [other]
        g_local.vertex_list.extend(axiom)
        g_local.coord_list = [v.coords for v in g_local.vertex_list]
        rebuild_tree(g_local)

        front = list(axiom)
        max_iter = params.get("max_iter", 200)
        it = 0
        while front and it < max_iter:
            newfront = []
            for vertex in front:
                b = 1.0
                suggestions = organic(vertex, b, params)
                for s in suggestions:
                    added = check(s, vertex, g_local, params)
                    if added is not None:
                        newfront.append(added)
            front = newfront
            it += 1

        def remove_shallow_angles(g_obj, threshold_deg=15):
            import math
            removals = set()
            idx_map = {v: i for i, v in enumerate(g_obj.vertex_list)}
            thr = math.radians(threshold_deg)
            for v in g_obj.vertex_list:
                neis = list(v.neighbours)
                if len(neis) < 2:
                    continue
                for i in range(len(neis)):
                    for j in range(i+1, len(neis)):
                        n1 = neis[i]
                        n2 = neis[j]
                        v1 = n1.coords - v.coords
                        v2 = n2.coords - v.coords
                        nrm1 = np.linalg.norm(v1)
                        nrm2 = np.linalg.norm(v2)
                        if nrm1 == 0 or nrm2 == 0:
                            continue
                        cosang = np.dot(v1, v2) / (nrm1 * nrm2)
                        cosang = max(min(cosang, 1.0), -1.0)
                        ang = math.acos(cosang)
                        if ang < thr:
                            if nrm1 < nrm2:
                                rem = frozenset({idx_map[v], idx_map[n1]})
                            else:
                                rem = frozenset({idx_map[v], idx_map[n2]})
                            removals.add(rem)
            removed_count = 0
            for rem in removals:
                if len(rem) != 2:
                    continue
                i1, i2 = tuple(rem)
                a = g_obj.vertex_list[i1]
                b = g_obj.vertex_list[i2]
                if b in a.neighbours:
                    a.neighbours.remove(b)
                if a in b.neighbours:
                    b.neighbours.remove(a)
                removed_count += 1
            return removed_count

        removed = remove_shallow_angles(g_local, threshold_deg=params.get("shallow_angle_deg", 30))
        if visualize and out_image is not None:
            fig, ax = plt.subplots(figsize=(8,8))
            for v in g_local.vertex_list:
                for n in v.neighbours:
                    xs = [v.coords[0], n.coords[0]]
                    ys = [v.coords[1], n.coords[1]]
                    ax.plot(xs, ys, color='black', linewidth=1)
            xs = [v.coords[0] for v in g_local.vertex_list]
            ys = [v.coords[1] for v in g_local.vertex_list]
            ax.scatter(xs, ys, color='red', s=8)
            ax.set_aspect('equal')
            ax.set_title(f"Simple organic roadmap — vertices: {len(g_local.vertex_list)}")
            plt.tight_layout()
            outp = out_image
            outp = os.path.join(os.path.dirname(__file__), outp)
            plt.savefig(outp, dpi=150)
            print(f"Saved visualization to {outp}")
        # Scale roadmap to meters (1:20)
        scale = params.get('scale', 20.0)
        for v in g_local.vertex_list:
            v.coords = v.coords * scale
        print(f"Total vertices: {len(g_local.vertex_list)} (removed {removed} shallow-angle edges)")
        return g_local

    # Place buildings and export scene
    assets_folder = os.path.join(os.path.dirname(__file__), 'data', 'building', 'obj')
    placement_params = {
        'placement_min_distance_to_road': params.get('placement_min_distance_to_road', 5.0),
        'placement_max_attempts': params.get('placement_max_attempts', 5000),
        'placement_max_buildings': params.get('placement_max_buildings', 40),
        'placement_scene_padding': params.get('placement_scene_padding', params.get('scene_padding', 2.0))
    }
    # Support generating multiple placement variants and export as a list of schemes
    placement_variants = int(params.get('placement_variants', 1))
    variants = []
    # Generate a base roadmap (visualization); this also ensures deterministic start if seed provided
    base_seed = params.get('seed', None)
    visualization_out = params.get("output_image", params.get("output", "simple_roadmap.png"))
    g = generate_road(params, seed=base_seed, visualize=True, out_image=visualization_out)

    car_assets_folder = os.path.join(os.path.dirname(__file__), 'data', 'car', 'obj')
    ped_assets_folder = os.path.join(os.path.dirname(__file__), 'data', 'ped', 'obj')

    car_params = {
        'car_max_distance_to_road': params.get('car_max_distance_to_road', 3.0),
        'car_max_attempts': params.get('car_max_attempts', 2000),
        'car_max_count': params.get('car_max_count', params.get('max_cars', 100)),
        'placement_scene_padding': params.get('placement_scene_padding', params.get('scene_padding', 2.0))
    }

    ped_params = {
        'ped_max_distance_to_road': params.get('ped_max_distance_to_road', 2.0),
        'ped_max_attempts': params.get('ped_max_attempts', 5000),
        'ped_max_count': params.get('ped_max_count', 50),
        'placement_scene_padding': params.get('placement_scene_padding', params.get('scene_padding', 2.0))
    }

    for vi in range(placement_variants):
        # For each variant, generate a fresh roadmap so placements are based on different road structures.
        if vi == 0:
            g_variant = g
        else:
            if base_seed is not None:
                variant_seed = int(base_seed) + vi
            else:
                variant_seed = None
            g_variant = generate_road(params, seed=variant_seed, visualize=False)
        b_records, b_meshes, b_rects = place_buildings(g_variant, assets_folder, placement_params)
        car_records, car_meshes, car_rects = place_cars(g_variant, car_assets_folder, car_params, existing_rects=b_rects)
        ped_records, ped_meshes, ped_rects = place_peds(g_variant, ped_assets_folder, ped_params, existing_rects=(b_rects + car_rects))

        combined_records = []
        combined_records.extend(b_records)
        combined_records.extend(car_records)
        combined_records.extend(ped_records)
        variants.append(combined_records)

    out_json = os.path.join(os.path.dirname(__file__), params.get('placement_output_json', os.path.join('data', 'scene', 'placements.json')))
    export_placement_json(variants, out_json)

    # visualization options
    visualize_scene = params.get('visualize_placement_scene', True)
    visualize_with_materials = params.get('visualize_with_materials', False)
    export_obj = params.get('export_placement_obj', False)

    if visualize_scene:
        visualize_placement_scene(out_json, use_materials=visualize_with_materials)
    if export_obj:
        out_obj = os.path.join(os.path.dirname(__file__), params.get('placement_output_obj', os.path.join('data', 'scene', 'placed_scene.obj')))
        generated = generate_obj_from_placement(out_json, out_obj)
        if generated:
            load_and_visualize_obj(generated)
    
def load_and_visualize_obj(obj_path: str):
    """Load an OBJ (with MTL) and visualize it interactively using trimesh.

    Also prints the bounding box min/max and dimensions.
    """
    if trimesh is None:
        print("trimesh is not installed. Install with `pip install trimesh pyglet` to enable OBJ visualization.")
        return
    if not os.path.exists(obj_path):
        print(f"OBJ file not found: {obj_path}")
        return
    # trimesh.load will return a Trimesh or a Scene depending on the file
    scene = trimesh.load(obj_path, force='scene')
    if isinstance(scene, trimesh.Scene):
        bounds = scene.bounds
        minb, maxb = bounds[0], bounds[1]
        dims = maxb - minb
        print(f"Scene bounds min: {minb}, max: {maxb}, dims: {dims}")
        # Show interactive window (requires pyglet and OpenGL)
        try:
            scene.show()
        except Exception as e:
            print("Failed to open trimesh scene viewer:", e)
    else:
        # single mesh
        mesh = scene
        minb, maxb = mesh.bounds[0], mesh.bounds[1]
        dims = maxb - minb
        print(f"Mesh bounds min: {minb}, max: {maxb}, dims: {dims}")
        try:
            mesh.show()
        except Exception as e:
            print("Failed to open trimesh mesh viewer:", e)


if __name__ == '__main__':
    params = {
        # Macro
        'placement_variants': 64,
        'seed': 114514,

        # Roadmap / growth parameters (canonical names)
        'border': (6, 6),
        'max_segment_length': 1.6,
        'min_node_distance': 0.7,
        'organic_length_min': 0.8,
        'organic_length_max': 1.6,
        'organic_prob_forward': 90,
        'organic_prob_turn': 12,
        'axiom': [[2,0],[3,0],[0,-2],[0,-3],[0,2],[0,3],[-2,0],[-3,0]],
        'max_iter': 200,
        # Output / visualization
        'output_image': 'simple_roadmap.png',
        # Postprocessing
        'shallow_angle_deg': 30,

        # Other
        'scale': 20.0,

        # Placement (buildings)
        'placement_min_distance_to_road': 5.0,
        'placement_max_attempts': 5000,
        'placement_max_buildings': 50 + random.randint(0, 20),
        'placement_scene_padding': 2.0,

        # Placement (cars)
        'car_max_attempts': 2000,
        'car_max_count': 50 + random.randint(0, 50),
        'car_max_distance_to_road': 3.0,

        # Pedestrians
        'ped_max_distance_to_road': 2.0,
        'ped_max_attempts': 5000,
        'ped_max_count': 100 + random.randint(0, 100),

        # Placement outputs (relative to package dir)
        'placement_output_json': os.path.join('data', 'scene', 'placements.json'),
        'placement_output_obj': os.path.join('data', 'scene', 'placed_scene.obj'),

        # Visualization
        'visualize_placement_scene': False,
        'visualize_with_materials': False,
        'export_placement_obj': False,
    }
    run(params)
