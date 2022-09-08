import open3d
import numpy as np
import sys

mesh = open3d.io.read_triangle_mesh(sys.argv[1])
mesh.compute_vertex_normals()
vis = open3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.get_render_option().background_color = np.array([0.6, 0.6, 0.6])
vis.get_render_option().mesh_show_wireframe = True
vis.get_render_option().light_on = False
vis.get_render_option().mesh_show_back_face = True
vis.run()
vis.destroy_window()
