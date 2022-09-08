from plyfile import PlyData, PlyElement
import glm
import numpy
import sys
import colorsys
import math
import time

from pathlib import Path

from helper_functions import ply_in
from helper_functions import ply_out
from helper_functions import icosphere_projection
from helper_functions import face_projection
from helper_functions import height_difference
from helper_functions import orientation_difference
from helper_functions import sampled_hausdorff_distance
from helper_functions import gen_colors
from helper_functions import generate_histogram
from helper_functions import text_out

def main(argv):
    start_time = time.time()
    # Prepare file information
    if len(argv) < 2:
        print("Subject number required")
        print("Usage: python {} subject_number [LOD]".format(argv[0]))
        exit(1)
    subj_num = int(argv[1])
    in_scan = "subjects/{0}/scan_{0}.ply".format(str(subj_num))
    in_avatar = "subjects/{0}/avatar_{0}.ply".format(str(subj_num))
    out_scan = "subjects/{0}/out_meshes/scan_{0}_proj.ply".format(str(subj_num))
    out_avatar = "subjects/{0}/out_meshes/avatar_{0}_proj.ply".format(str(subj_num))
    out_scan_hd = "subjects/{0}/out_meshes/scan_{0}_hd.ply".format(str(subj_num))
    out_avatar_hd = "subjects/{0}/out_meshes/avatar_{0}_hd.ply".format(str(subj_num))
    out_scan_od = "subjects/{0}/out_meshes/scan_{0}_od.ply".format(str(subj_num))
    out_avatar_od = "subjects/{0}/out_meshes/avatar_{0}_od.ply".format(str(subj_num))
    out_avatar_shd = "subjects/{0}/out_meshes/avatar_{0}_shd.ply".format(str(subj_num))
    face_out_scan = "subjects/{0}/out_meshes/face_scan_{0}_proj.ply".format(str(subj_num))
    face_out_avatar = "subjects/{0}/out_meshes/face_avatar_{0}_proj.ply".format(str(subj_num))
    face_out_scan_hd = "subjects/{0}/out_meshes/face_scan_{0}_hd.ply".format(str(subj_num))
    face_out_avatar_hd = "subjects/{0}/out_meshes/face_avatar_{0}_hd.ply".format(str(subj_num))
    face_out_scan_od = "subjects/{0}/out_meshes/face_scan_{0}_od.ply".format(str(subj_num))
    face_out_avatar_od = "subjects/{0}/out_meshes/face_avatar_{0}_od.ply".format(str(subj_num))
    face_out_avatar_shd = "subjects/{0}/out_meshes/face_avatar_{0}_shd.ply".format(str(subj_num))
    lod = 5
    if len(argv) >= 3:
        lod = int(argv[2])
    
    # Load the mesh data
    vertices_scan, faces_scan = ply_in(in_scan)
    vertices_avatar, faces_avatar = ply_in(in_avatar)
    print("Meshes successfully read...")

    # Do icosphere projection on meshes
    
    print("Performing mesh projection on scan...")
    proj_vertices_scan, proj_faces_scan = icosphere_projection(vertices_scan, faces_scan, lod)
    print("Performing mesh projection on avatar...")
    proj_vertices_avatar, proj_faces_avatar = icosphere_projection(vertices_avatar, faces_avatar, lod)
    

    """
    print("Performing mesh projection on scan...")
    proj_vertices_scan, proj_faces_scan = face_projection(vertices_scan, faces_scan, 7)
    print("Performing mesh projection on avatar...")
    proj_vertices_avatar, proj_faces_avatar = face_projection(vertices_avatar, faces_avatar, 7)
    """

    # Do file output of intermediate files
    Path("subjects/{}/out_meshes".format(str(subj_num))).mkdir(parents=True, exist_ok=True)
    ply_out(proj_vertices_scan, proj_faces_scan, filename=out_scan)
    ply_out(proj_vertices_avatar, proj_faces_avatar, filename=out_avatar)
    print("Projected mesh output complete...")

    # Do mesh comparison
    print("Performing mesh comparison...")
    mesh_hd = height_difference(proj_vertices_scan, proj_vertices_avatar)
    mesh_od = orientation_difference(proj_vertices_scan, proj_vertices_avatar, proj_faces_avatar)
    mesh_shd = sampled_hausdorff_distance(proj_vertices_avatar, proj_vertices_scan, vertices_scan, faces_scan)
    print("Mesh comparison complete...")

    # Generate colors for output
    hd_colors = gen_colors(mesh_hd, low_bound = 0.0015, up_bound=0.020)
    od_colors = gen_colors(mesh_od)
    shd_colors = gen_colors(mesh_shd, low_bound = 0.0015, up_bound=0.020)

    # Output meshes with colors
    ply_out(proj_vertices_scan, proj_faces_scan, colors=hd_colors, filename=out_scan_hd)
    ply_out(proj_vertices_avatar, proj_faces_avatar, colors=hd_colors, filename=out_avatar_hd)
    ply_out(proj_vertices_scan, proj_faces_scan, colors=od_colors, filename=out_scan_od)
    ply_out(proj_vertices_avatar, proj_faces_avatar, colors=od_colors, filename=out_avatar_od)
    ply_out(proj_vertices_avatar, proj_faces_avatar, colors=shd_colors, filename=out_avatar_shd)
    print("Meshes with difference output complete...")

    # Output histograms
    Path("subjects/{}/graphs".format(str(subj_num))).mkdir(parents=True, exist_ok=True)
    generate_histogram(mesh_hd, 100, "Height Difference (mm) - Subject {}".format(str(subj_num)), "Height Difference (mm)", "subjects/{0}/graphs/hd_{0}.png".format(str(subj_num)), 1000.0)
    generate_histogram(mesh_od, 100, "Orientation Difference (degrees) - Subject {}".format(str(subj_num)),"Orientation Difference (degrees)", "subjects/{0}/graphs/od_{0}.png".format(str(subj_num)))
    generate_histogram(mesh_shd, 100, "Sampled Hausdorff Distance (mm) - Subject {}".format(str(subj_num)), "Sampled Hausdorff Distance (mm)", "subjects/{0}/graphs/shd_{0}.png".format(str(subj_num)), 1000.0)

    # Do textual output
    text_out(mesh_hd, mesh_od, mesh_shd, "subjects/{0}/stats_{0}.txt".format(str(subj_num)))


    # Print final runtime for sanity
    print("Runtime: {:.3f} seconds".format(time.time() - start_time))


main(sys.argv)