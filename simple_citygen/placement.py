import os
import random
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix
_ASSET_DIM_CACHE = {}
_MESH_CACHE = {}
_MESH_CACHE_WITH_MATERIALS = {}

try:
    from .geometry import nearest_edge
except Exception:
    from geometry import nearest_edge


def extract_segments(g, scale=1.0):
    """Return list of unique undirected edges (a,b) from graph g.

    This does not modify g; it expects vertex.coords are already scaled if needed.
    """
    seen = set()
    segs = []
    for a in g.vertex_list:
        for b in a.neighbours:
            eid = tuple(sorted((id(a), id(b))))
            if eid in seen:
                continue
            seen.add(eid)
            segs.append((a, b))
    return segs


def load_building_assets(folder):
    assets = []
    if trimesh is None:
        print("trimesh not available; cannot load assets")
        return assets
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(folder, fn)
        try:
            if path in _ASSET_DIM_CACHE:
                w, h = _ASSET_DIM_CACHE[path]
            else:
                mesh = trimesh.load(path, force='mesh', skip_materials=True)
                if mesh.is_empty:
                    continue
                R = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                mesh.apply_transform(R)
                minb, maxb = mesh.bounds
                w = float(maxb[0] - minb[0])
                h = float(maxb[1] - minb[1])
                _ASSET_DIM_CACHE[path] = (w, h)
            assets.append({'name': fn, 'path': path, 'w': w, 'h': h})
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return assets


def rect_corners(center, w, h, theta):
    """Return 4 corner points of rectangle centered at center (2D) rotated by theta."""
    cx, cy = center
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = (R @ corners.T).T + np.array([cx, cy])
    return pts


def polygons_overlap(poly1, poly2):
    """SAT test for convex polygons (2D arrays of points in order)."""
    def proj(poly, axis):
        dots = poly @ axis
        return dots.min(), dots.max()

    polys = [np.asarray(poly1), np.asarray(poly2)]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i+1) % n]
            edge = b - a
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm == 0:
                continue
            axis = axis / norm
            min1, max1 = proj(poly1, axis)
            min2, max2 = proj(poly2, axis)
            if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
                return False
    return True


def place_buildings(g, assets_folder, params, keep_transformed_meshes=False):
    """Place buildings on scaled roadmap g and return list of placement records.

    Each placement record is a dict:
        { 'src': <path to obj>, 'position': [x,y,z], 'rotation': theta_in_radians }

    This function no longer returns transformed meshes to simplify later export.
    """
    if trimesh is None:
        raise RuntimeError('trimesh is required for placement')

    assets = load_building_assets(assets_folder)
    if not assets:
        print('No assets loaded; abort placement')
        return []

    # scene bounds from vertex coords
    coords = np.array([v.coords for v in g.vertex_list])
    min_xy = coords.min(axis=0)[:2]
    max_xy = coords.max(axis=0)[:2]
    padding = params.get('placement_scene_padding', 2.0)
    min_x, min_y = min_xy - padding
    max_x, max_y = max_xy + padding

    dis_min_bld2rd = params.get('placement_min_distance_to_road', 5.0)
    max_attempts = params.get('placement_max_attempts', 500)
    max_buildings = params.get('placement_max_buildings', 50)

    placed_meshes = []
    placed_rects = []
    attempts = 0
    placed_count = 0
    placed_records = []

    while attempts < max_attempts and placed_count < max_buildings:
        attempts += 1
        asset = random.choice(assets)
        # random center
        px = random.uniform(min_x, max_x)
        py = random.uniform(min_y, max_y)
        p = np.array([px, py])
        dist, edge_pair, closest_point, t = nearest_edge(p, g)
        if edge_pair[0] is None:
            continue
        if dist < dis_min_bld2rd:
            continue
        a, b = edge_pair
        dirv = b.coords - a.coords
        theta = np.arctan2(dirv[1], dirv[0])

        rect = rect_corners((px, py), asset['w'], asset['h'], theta)
        # check corners distance to roads
        corner_dists = [nearest_edge(pt, g)[0] for pt in rect]
        if min(corner_dists) < dis_min_bld2rd:
            continue
        # check overlap with existing
        overlap = False
        for other in placed_rects:
            if polygons_overlap(rect, other):
                overlap = True
                break
        if overlap:
            continue

        urdf_path = asset['path'].replace(os.sep + 'obj' + os.sep, os.sep + 'urdf' + os.sep)
        urdf_path = os.path.splitext(urdf_path)[0] + '.urdf'
        src_path = urdf_path if os.path.exists(urdf_path) else asset['path']
        placed_records.append({'src': src_path, 'obj_src': asset['path'], 'position': [float(px), float(py), 0.0], 'rotation': float(theta)})
        if keep_transformed_meshes:
            try:
                if asset['path'] in _MESH_CACHE_WITH_MATERIALS:
                    base = _MESH_CACHE_WITH_MATERIALS[asset['path']]
                else:
                    base = trimesh.load(asset['path'], force='mesh', skip_materials=False)
                    R_up = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                    base.apply_transform(R_up)
                    _MESH_CACHE_WITH_MATERIALS[asset['path']] = base
                mesh = base.copy()
                minb, maxb = mesh.bounds
                mesh_center = (minb + maxb) / 2.0
                T_translate_to_origin = translation_matrix(-mesh_center)
                Rz = rotation_matrix(theta, [0, 0, 1])
                T_translate_to_pos = translation_matrix([px, py, -minb[2] + mesh_center[2]])
                transform = T_translate_to_pos.dot(Rz).dot(T_translate_to_origin)
                mesh.apply_transform(transform)
                placed_meshes.append(mesh)
            except Exception:
                pass
        placed_rects.append(rect)
        placed_count += 1

    print(f"Placement: attempts={attempts}, placed={placed_count}")
    # return records, meshes and rects
    return placed_records, placed_meshes, placed_rects


def load_vehicle_assets(folder):
    assets = []
    if trimesh is None:
        print("trimesh not available; cannot load vehicle assets")
        return assets
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(folder, fn)
        try:
            if path in _ASSET_DIM_CACHE:
                w, h = _ASSET_DIM_CACHE[path]
            else:
                mesh = trimesh.load(path, force='mesh', skip_materials=True)
                if mesh.is_empty:
                    continue
                R = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                try:
                    mesh.apply_transform(R)
                except Exception:
                    pass
                minb, maxb = mesh.bounds
                w = float(maxb[0] - minb[0])
                h = float(maxb[1] - minb[1])
                _ASSET_DIM_CACHE[path] = (w, h)
            assets.append({'name': fn, 'path': path, 'w': w, 'h': h})
        except Exception as e:
            print(f"Failed to load vehicle {path}: {e}")
    return assets


def place_cars(g, assets_folder, params, existing_rects=None, keep_transformed_meshes=False):
    """Place cars alongside roads.

    Rules:
      - Car direction parallel to nearest road segment
      - Car center distance to road <= dis_max_to_road (e.g., 3m)
      - Avoid overlap with existing_rects (buildings) and other cars

    Returns (car_records, car_meshes, car_rects)
    """
    if trimesh is None:
        raise RuntimeError('trimesh is required for car placement')

    assets = load_vehicle_assets(assets_folder)
    if not assets:
        print('No vehicle assets loaded; abort car placement')
        return [], [], []

    # scene bounds from vertex coords
    coords = np.array([v.coords for v in g.vertex_list])
    min_xy = coords.min(axis=0)[:2]
    max_xy = coords.max(axis=0)[:2]
    padding = params.get('placement_scene_padding', 2.0)
    min_x, min_y = min_xy - padding
    max_x, max_y = max_xy + padding

    dis_max = params.get('car_max_distance_to_road', 3.0)
    max_attempts = params.get('car_max_attempts', 1000)
    max_cars = params.get('car_max_count', 100)

    segs = extract_segments(g)

    car_meshes = []
    car_rects = []
    car_records = []
    attempts = 0
    placed = 0

    while attempts < max_attempts and placed < max_cars:
        attempts += 1
        asset = random.choice(assets)
        # sample random segment and position along it
        if not segs:
            break
        a, b = random.choice(segs)
        t = random.random()
        seg_pt = a.coords + t * (b.coords - a.coords)
        # compute normal (unit)
        v = b.coords - a.coords
        n = np.array([-v[1], v[0]])
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            continue
        n = n / n_norm
        # offset from segment within -dis_max..dis_max
        offset = random.uniform(-dis_max, dis_max)
        center_xy = seg_pt[:2] + n * offset
        # ensure within bounds
        if center_xy[0] < min_x or center_xy[0] > max_x or center_xy[1] < min_y or center_xy[1] > max_y:
            continue
        # orientation parallel to segment
        theta = np.arctan2(v[1], v[0])

        rect = rect_corners(center_xy, asset['w'], asset['h'], theta)
        # check overlap with existing objects
        overlap = False
        if existing_rects:
            for other in existing_rects:
                if polygons_overlap(rect, other):
                    overlap = True
                    break
        if overlap:
            continue
        for other in car_rects:
            if polygons_overlap(rect, other):
                overlap = True
                break
        if overlap:
            continue

        px, py = float(center_xy[0]), float(center_xy[1])
        urdf_path = asset['path'].replace(os.sep + 'obj' + os.sep, os.sep + 'urdf' + os.sep)
        urdf_path = os.path.splitext(urdf_path)[0] + '.urdf'
        src_path = urdf_path if os.path.exists(urdf_path) else asset['path']
        car_records.append({'src': src_path, 'obj_src': asset['path'], 'position': [px, py, 0.0], 'rotation': float(theta)})
        if keep_transformed_meshes:
            try:
                if asset['path'] in _MESH_CACHE_WITH_MATERIALS:
                    base = _MESH_CACHE_WITH_MATERIALS[asset['path']]
                else:
                    base = trimesh.load(asset['path'], force='mesh', skip_materials=False)
                    R_up = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                    try:
                        base.apply_transform(R_up)
                    except Exception:
                        pass
                    _MESH_CACHE_WITH_MATERIALS[asset['path']] = base
                mesh = base.copy()
                minb, maxb = mesh.bounds
                mesh_center = (minb + maxb) / 2.0
                T_translate_to_origin = translation_matrix(-mesh_center)
                Rz = rotation_matrix(theta, [0, 0, 1])
                T_translate_to_pos = translation_matrix([px, py, -minb[2] + mesh_center[2]])
                transform = T_translate_to_pos.dot(Rz).dot(T_translate_to_origin)
                mesh.apply_transform(transform)
                car_meshes.append(mesh)
            except Exception:
                pass
        car_rects.append(rect)
        placed += 1

    print(f"Car placement: attempts={attempts}, placed={placed}")
    return car_records, car_meshes, car_rects


def load_ped_assets(folder):
    assets = []
    if trimesh is None:
        print("trimesh not available; cannot load pedestrian assets")
        return assets
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(folder, fn)
        try:
            if path in _ASSET_DIM_CACHE:
                w, h = _ASSET_DIM_CACHE[path]
            else:
                mesh = trimesh.load(path, force='mesh', skip_materials=True)
                if mesh.is_empty:
                    continue
                R = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                try:
                    mesh.apply_transform(R)
                except Exception:
                    pass
                minb, maxb = mesh.bounds
                w = float(maxb[0] - minb[0])
                h = float(maxb[1] - minb[1])
                _ASSET_DIM_CACHE[path] = (w, h)
            assets.append({'name': fn, 'path': path, 'w': w, 'h': h})
        except Exception as e:
            print(f"Failed to load pedestrian {path}: {e}")
    return assets


def place_peds(g, assets_folder, params, existing_rects=None, keep_transformed_meshes=False):
    if trimesh is None:
        raise RuntimeError('trimesh is required for pedestrian placement')
    assets = load_ped_assets(assets_folder)
    if not assets:
        print('No pedestrian assets loaded; abort pedestrian placement')
        return [], [], []

    coords = np.array([v.coords for v in g.vertex_list])
    min_xy = coords.min(axis=0)[:2]
    max_xy = coords.max(axis=0)[:2]
    padding = params.get('placement_scene_padding', 2.0)
    min_x, min_y = min_xy - padding
    max_x, max_y = max_xy + padding

    max_attempts = params.get('ped_max_attempts', 5000)
    max_peds = params.get('ped_max_count', 50)

    ped_meshes = []
    ped_rects = []
    ped_records = []
    attempts = 0
    placed = 0

    # simplified: place pedestrians anywhere in scene bounds as long as they don't overlap others
    while attempts < max_attempts and placed < max_peds:
        attempts += 1
        asset = random.choice(assets)

        # random center anywhere in bounds (with padding)
        px = random.uniform(min_x, max_x)
        py = random.uniform(min_y, max_y)
        center_xy = np.array([px, py])

        theta = random.uniform(0.0, 2.0 * np.pi)

        rect = rect_corners(center_xy, asset['w'], asset['h'], theta)

        # check overlap with existing objects
        overlap = False
        if existing_rects:
            for other in existing_rects:
                if polygons_overlap(rect, other):
                    overlap = True
                    break
        if overlap:
            continue
        for other in ped_rects:
            if polygons_overlap(rect, other):
                overlap = True
                break
        if overlap:
            continue

        px_f, py_f = float(px), float(py)
        urdf_path = asset['path'].replace(os.sep + 'obj' + os.sep, os.sep + 'urdf' + os.sep)
        urdf_path = os.path.splitext(urdf_path)[0] + '.urdf'
        src_path = urdf_path if os.path.exists(urdf_path) else asset['path']
        ped_records.append({'src': src_path, 'obj_src': asset['path'], 'position': [px_f, py_f, 0.0], 'rotation': float(theta)})

        if keep_transformed_meshes:
            try:
                if asset['path'] in _MESH_CACHE_WITH_MATERIALS:
                    base = _MESH_CACHE_WITH_MATERIALS[asset['path']]
                else:
                    base = trimesh.load(asset['path'], force='mesh', skip_materials=False)
                    R_up = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                    try:
                        base.apply_transform(R_up)
                    except Exception:
                        pass
                    _MESH_CACHE_WITH_MATERIALS[asset['path']] = base
                mesh = base.copy()
                minb, maxb = mesh.bounds
                mesh_center = (minb + maxb) / 2.0
                T_translate_to_origin = translation_matrix(-mesh_center)
                Rz = rotation_matrix(theta, [0, 0, 1])
                T_translate_to_pos = translation_matrix([px_f, py_f, -minb[2] + mesh_center[2]])
                transform = T_translate_to_pos.dot(Rz).dot(T_translate_to_origin)
                mesh.apply_transform(transform)
                ped_meshes.append(mesh)
            except Exception:
                pass

        ped_rects.append(rect)
        placed += 1

    print(f"Pedestrian placement: attempts={attempts}, placed={placed}")
    return ped_records, ped_meshes, ped_rects


def export_scene(mesh_list, out_path):
    if trimesh is None:
        raise RuntimeError('trimesh is required to export scene')
    if not mesh_list:
        print('No meshes to export')
        return None
    scene = trimesh.Scene()
    for i, m in enumerate(mesh_list):
        name = f'obj_{i}'
        scene.add_geometry(m, node_name=name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        scene.export(out_path)
        print(f'Exported scene to {out_path}')
        return out_path
    except Exception as e:
        print('Failed to export scene:', e)
        return None


def export_placement_json(placements, out_json_path):
    """Write placement records to JSON file."""
    import json
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(placements, f, indent=2)
    print(f'Wrote placement JSON to {out_json_path}')
    return out_json_path


def generate_obj_from_placement(json_path, out_obj_path):
    import json
    if trimesh is None:
        raise RuntimeError('trimesh required to generate OBJ from JSON')
    with open(json_path, 'r') as f:
        placements = json.load(f)
    scene = trimesh.Scene()
    count = 0
    for rec in placements:
        src = rec.get('obj_src', rec['src'])
        pos = rec['position']
        theta = rec['rotation']
        try:
            if src in _MESH_CACHE_WITH_MATERIALS:
                base = _MESH_CACHE_WITH_MATERIALS[src]
            else:
                base = trimesh.load(src, force='mesh', skip_materials=False)
                R_up = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                base.apply_transform(R_up)
                _MESH_CACHE_WITH_MATERIALS[src] = base
            minb, maxb = base.bounds
            mesh_center = (minb + maxb) / 2.0
            T_translate_to_origin = translation_matrix(-mesh_center)
            Rz = rotation_matrix(theta, [0, 0, 1])
            tz = pos[2] + (-minb[2] + mesh_center[2])
            T_translate_to_pos = translation_matrix([pos[0], pos[1], tz])
            transform = T_translate_to_pos.dot(Rz).dot(T_translate_to_origin)
            geom_name = f"geom::{src}"
            scene.add_geometry(base, node_name=f'obj_{count}', geom_name=geom_name, transform=transform)
            count += 1
        except Exception as e:
            print(f'Failed to load/transform {src}: {e}')
    if count == 0:
        print('No meshes to export from placement JSON')
        return None
    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    try:
        scene.export(out_obj_path)
        print(f'Generated OBJ from JSON: {out_obj_path}')
        return out_obj_path
    except Exception as e:
        print('Failed to export generated OBJ:', e)
        return None


def build_scene_from_placement(json_path, use_materials=False):
    import json
    if trimesh is None:
        raise RuntimeError('trimesh required to build scene from JSON')
    with open(json_path, 'r') as f:
        placements = json.load(f)
    scene = trimesh.Scene()
    count = 0
    for rec in placements:
        src = rec.get('obj_src', rec['src'])
        pos = rec['position']
        theta = rec['rotation']
        try:
            cache = _MESH_CACHE_WITH_MATERIALS if use_materials else _MESH_CACHE
            if src in cache:
                base = cache[src]
            else:
                base = trimesh.load(src, force='mesh', skip_materials=(not use_materials))
                R_up = rotation_matrix(np.deg2rad(90), [1, 0, 0])
                base.apply_transform(R_up)
                cache[src] = base
            minb, maxb = base.bounds
            mesh_center = (minb + maxb) / 2.0
            T_translate_to_origin = translation_matrix(-mesh_center)
            Rz = rotation_matrix(theta, [0, 0, 1])
            tz = pos[2] + (-minb[2] + mesh_center[2])
            T_translate_to_pos = translation_matrix([pos[0], pos[1], tz])
            transform = T_translate_to_pos.dot(Rz).dot(T_translate_to_origin)
            geom_name = f"geom::{src}"
            scene.add_geometry(base, node_name=f'obj_{count}', geom_name=geom_name, transform=transform)
            count += 1
        except Exception as e:
            print(f'Failed to add {src} to scene: {e}')
    if count == 0:
        print('No nodes added to scene from placement JSON')
    return scene


def visualize_placement_scene(json_path, use_materials=False):
    scene = build_scene_from_placement(json_path, use_materials=use_materials)
    try:
        scene.show()
    except Exception as e:
        print('Failed to open scene viewer:', e)
