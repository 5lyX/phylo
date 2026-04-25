import open3d as o3d
import numpy as np
from constants import GEOM_FIXED_SOURCES_PATH

# read STL file
stl_file = f"{GEOM_FIXED_SOURCES_PATH}/round_bowl.stl"  # change to your file
mesh = o3d.io.read_triangle_mesh(stl_file)

# compute normals
if not mesh.has_triangle_normals():
    print("no triangle normals, will compute first")
    mesh.compute_triangle_normals()
if not mesh.has_vertex_normals():
    print("no vertex normals, will compute first")
    mesh.compute_vertex_normals()

# set mesh color 
mesh.paint_uniform_color([0.8, 0.8, 0.8])

# get edges from triangles
triangles = np.asarray(mesh.triangles)
edges = set()
for tri in triangles:
    edges.add(tuple(sorted([tri[0], tri[1]])))
    edges.add(tuple(sorted([tri[1], tri[2]])))
    edges.add(tuple(sorted([tri[2], tri[0]])))
edges = list(edges)

# create line set
lines = o3d.geometry.LineSet(
    points=mesh.vertices,  # vertex
    lines=o3d.utility.Vector2iVector(edges)
)
lines.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in edges])

# get triangle centers and normals
triangle_normals = np.asarray(mesh.triangle_normals)
triangle_centers = np.mean(np.asarray(mesh.vertices)[triangles], axis=1)

# create arrow lines
normal_lines = []
arrow_points = []
for i, center in enumerate(triangle_centers):
    arrow_points.append(center)  # start
    arrow_points.append(center + 0.1 * triangle_normals[i])  # end
    normal_lines.append([len(arrow_points) - 2, len(arrow_points) - 1])

arrow_lines = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(arrow_points),
    lines=o3d.utility.Vector2iVector(normal_lines),
)
arrow_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in normal_lines])  # red

# visualize
o3d.visualization.draw_geometries(
    [mesh, lines, arrow_lines],
    window_name="Mesh Normals",
    width=800,
    height=600,
    mesh_show_back_face=True
)