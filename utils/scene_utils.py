import os
import trimesh


def load_all_obj_meshes(mesh_dir: str):
    paths = [
        os.path.join(mesh_dir, fn)
        for fn in os.listdir(mesh_dir)
        if ".obj" in fn
    ]
    paths.sort()

    meshes = []
    mesh_indices = []
    centers = {}
    for idx, p in enumerate(paths):
        tm = trimesh.load(p, force="scene", process=False)
        c = tm.bounding_box_oriented.centroid  # raw coords
        centers[idx] = c

        if isinstance(tm, trimesh.Scene):
            dumped = tm.dump()
            for g in dumped:
                if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0:
                    meshes.append(g)
                    mesh_indices.append(idx)
        elif isinstance(tm, trimesh.Trimesh) and len(tm.vertices) > 0:
            meshes.append(tm)
            mesh_indices.append(idx)

    if len(meshes) == 0:
        raise RuntimeError("No OBJ meshes found in mesh_dir.")
    return meshes, mesh_indices, centers