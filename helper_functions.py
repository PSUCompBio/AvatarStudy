from plyfile import PlyData, PlyElement
import glm
import numpy
import sys
import colorsys
import math
import time
import matplotlib.pyplot as plt

# Read in a PLY file given a file name
def ply_in(filename):
    # Read in the raw data
    try:
        with open(filename, 'rb') as file:
            data = PlyData.read(file)
    except FileNotFoundError:
        print("File", filename, "does not exist")
        exit(1)

    # Extract vertices and faces from raw data
    vertices = []
    faces = []
    for i in range(len(data['vertex'])):
        # Get vertices as glm vec3 type for easier use later
        tempcoord = glm.vec3(data['vertex'][i]['x'], data['vertex'][i]['y'], data['vertex'][i]['z'])
        vertices.append(tempcoord)
    for i in range(len(data['face'])):
        tempface = (data['face'][i][0][0], data['face'][i][0][1], data['face'][i][0][2])
        faces.append(tempface)

    # Return the values
    return (vertices, faces)


# Write out a PLY file given vertices and faces
# Colors and file name are optional
def ply_out(vertices, faces, colors=None, filename='sample.ply'):
    # Prepare vertices
    out_verts = []
    for vertex in vertices:
        temp_vert = (vertex[0], vertex[1], vertex[2])
        out_verts.append(temp_vert)
    out_vertex = numpy.array(out_verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # prepare Faces
    out_faces = []
    for i in range(len(faces)):
        temp_points = [faces[i][0], faces[i][1], faces[i][2]]
        temp_face = tuple([temp_points])
        out_faces.append(temp_face)
    out_face = numpy.array(out_faces, dtype=[('vertex_indices', 'i4', (3,))])

    # Prepare colors if needed
    if colors != None:
        out_colors = []
        for color in colors:
            temp_color = (int(color[0]), int(color[1]), int(color[2]))
            out_colors.append(temp_color)
        out_color = numpy.array(out_colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Final preparation then output
    if colors == None:
        el1 = PlyElement.describe(out_vertex, 'vertex')
    else:
        try:
            size = len(out_vertex)
            assert len(out_color) == size
            out_combined = numpy.empty(size, out_vertex.dtype.descr + out_color.dtype.descr)
            for prop in out_vertex.dtype.names:
                out_combined[prop] = out_vertex[prop]
            for prop in out_color.dtype.names:
                out_combined[prop] = out_color[prop]
            el1 = PlyElement.describe(out_combined, 'vertex')
        except AssertionError:
            print("ERROR: Color array does not have same size as vertex array")
            print("Could not output", filename)
            exit(1)
    el2 = PlyElement.describe(out_face, 'face')
    PlyData([el1, el2], text=True).write(filename)


# Generate triangles from vertices and faces
def gen_tris(mesh_verts, mesh_faces):
    triangles = []
    for face in mesh_faces:
        temptri = (mesh_verts[face[0]], mesh_verts[face[1]], mesh_verts[face[2]])
        triangles.append(temptri)
    return triangles


# Generate vertex colors based on error function
def gen_colors(err_vals, low_bound = None, up_bound = None):
    # Base values
    err_vals_filtered = [val for val in err_vals if not isinstance(val, str)]
    if low_bound == None:
        err_min = min(err_vals_filtered)
    else:
        err_min = low_bound
    if up_bound == None:
        err_max = max(err_vals_filtered)
    else:
        err_max = up_bound
    err_dif = err_max - err_min
    
    # Compute colors
    out_colors = []
    for err in err_vals:
        if err == "ERROR":
            temp_hue = 300
        elif err < err_min:
            temp_hue = 240
        elif err > err_max:
            temp_hue = 0
        elif err_dif == 0.0:
            temp_hue = 240
        else:
            temp_hue = 240 - int(240 * ((err - err_min) / err_dif))
        temp_hsv = (temp_hue, 1, 1)
        temp_color = hsv_to_rgb_wrap(temp_hsv)
        out_colors.append(temp_color)
    
    return out_colors


# Wrapper function to get HSV to RGB in proper format [0, 255]
def hsv_to_rgb_wrap(in_hsv):
    H = float(in_hsv[0]) / 360.0
    S = float(in_hsv[1])
    V = float(in_hsv[2])
    out_rgb = colorsys.hsv_to_rgb(H, S, V)
    return (int(255 * out_rgb[0]), int(255 * out_rgb[1]), int(255 * out_rgb[2]))


# Perform triangle intersection for a single triangle
def tri_intersect(triangle, start, direction):
    # Make sure direction is normalized
    direction = glm.normalize(direction)

    # Extract points from triangle
    point0 = triangle[0]
    point1 = triangle[1]
    point2 = triangle[2]

    # Create variables for implementation of Cramer's rule
    a = point0.x - point1.x
    b = point0.y - point1.y
    c = point0.z - point1.z
    d = point0.x - point2.x
    e = point0.y - point2.y
    f = point0.z - point2.z
    g = direction.x
    h = direction.y
    i = direction.z
    j = point0.x - start.x
    k = point0.y - start.y
    l = point0.z - start.z

    # Create reusable variables for Cramer's rule
    ei_hf = e * i - h * f
    gf_di = g * f - d * i
    dh_eg = d * h - e * g
    ak_jb = a * k - j * b
    jc_al = j * c - a * l
    bl_kc = b * l - k * c

    # Solve values of beta and gamma
    M = a * ei_hf + b * gf_di + c * dh_eg
    if (M == 0.0):
        return 9999999
    beta = (j * ei_hf + k * gf_di + l * dh_eg) / M
    gamma = (i * ak_jb + h * jc_al + g * bl_kc) / M

    # Intersection detected
    if (beta >= 0.0 and gamma >= 0.0 and beta + gamma <= 1.0):
        t = -1.0 * (f * ak_jb + e * jc_al + d * bl_kc) / M
        if (t >= 0.00001):
            return t
    # No valid intersection detected
    return 9999999


# Perform triangle intersection for all triangles
def ray_intersect(tri_list, start, direction, max=True):
    # Set initial max/min values
    max_val = 0.0
    min_val = 9999999.0
    intersect_found = False

    # Iterate through all triangles, updating max and min
    for tri in tri_list:
        intersect_val = tri_intersect(tri, start, direction)
        if (intersect_val == 9999999):
            continue
        intersect_found = True
        if (intersect_val > max_val):
            max_val = intersect_val
        if (intersect_val < min_val):
            min_val = intersect_val

    # If no intersection found, return 9999999
    if (not intersect_found):
        return 9999999

    # If using max, return max val
    if (max):
        return max_val

    # Otherwise, return min val
    return min_val


# Perform projection using icosphere of given subdivision
def icosphere_projection(mesh_verts, mesh_faces, subdivs):
    # Read specified icosphere
    filename = "icospheres/icosphere{}.ply".format(subdivs)
    ico_mesh = ply_in(filename)
    ico_verts = ico_mesh[0]
    ico_faces = ico_mesh[1]
    
    # Initialize mesh triangles
    mesh_tris = gen_tris(mesh_verts, mesh_faces)

    # Perform the projection
    proj_verts = []
    completed_iterations = 0
    total_iterations = len(ico_verts)
    for vert in ico_verts:
        proj_val = ray_intersect(mesh_tris, glm.vec3(0.0, 0.0, 0.0), vert)
        if (proj_val == 9999999):
            proj_verts.append(glm.vec3(0.0, 0.0, 0.0))
        else:
            proj_verts.append(proj_val * vert)
        completed_iterations += 1
        if (completed_iterations % 25 == 0):
            print_progress(completed_iterations, total_iterations)
    print_progress(completed_iterations, total_iterations)
    proj_verts.append(glm.vec3(0.0, 0.0, 0.0))
    proj_verts.append(glm.vec3(0.0, 0.0, 0.0))
    proj_verts.append(glm.vec3(0.0, 0.0, 0.0))

    # Index the faces
    proj_faces = []
    error_face = (len(proj_verts) - 1, len(proj_verts) - 2, len(proj_verts) - 3)
    for face in ico_faces:
        if (is_zero(proj_verts[face[0]]) or is_zero(proj_verts[face[1]]) or is_zero(proj_verts[face[2]])):
            proj_faces.append(error_face)
        else:
            proj_faces.append(face)

    return (proj_verts, proj_faces)


# Perform projection using icosphere of given subdivision
def face_projection(mesh_verts, mesh_faces, subdivs):
    # Read specified icosphere
    filename = "icospheres/face{}.ply".format(subdivs)
    ico_mesh = ply_in(filename)
    ico_verts = ico_mesh[0]
    ico_faces = ico_mesh[1]
    
    # Initialize mesh triangles
    mesh_tris = gen_tris(mesh_verts, mesh_faces)

    # Perform the projection
    proj_verts = []
    completed_iterations = 0
    total_iterations = len(ico_verts)
    for vert in ico_verts:
        proj_val = ray_intersect(mesh_tris, glm.vec3(0.0, 0.0, 0.0), vert)
        if (proj_val == 9999999):
            proj_verts.append(glm.vec3(0.0, 0.0, 0.0))
        else:
            proj_verts.append(proj_val * vert)
        completed_iterations += 1
        if (completed_iterations % 25 == 0):
            print_progress(completed_iterations, total_iterations)
    print_progress(completed_iterations, total_iterations)
    proj_verts.append(glm.vec3(0.0, 0.0, 0.0))
    proj_verts.append(glm.vec3(0.0, 0.0, 0.0))
    proj_verts.append(glm.vec3(0.0, 0.0, 0.0))

    # Index the faces
    proj_faces = []
    error_face = (len(proj_verts) - 1, len(proj_verts) - 2, len(proj_verts) - 3)
    for face in ico_faces:
        if (is_zero(proj_verts[face[0]]) or is_zero(proj_verts[face[1]]) or is_zero(proj_verts[face[2]])):
            proj_faces.append(error_face)
        else:
            proj_faces.append(face)

    return (proj_verts, proj_faces)


# Helper function for seeing if a vector is a zero vector
def is_zero(vector_in):
    return (vector_in.x == 0.0 and vector_in.y == 0.0 and vector_in.z == 0)


# Prints a progress bar to the console
def print_progress (iteration, total, fill = '█', printEnd = "\r"):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(50 * iteration // total)
    bar = fill * filled_length + '-' * (50 - filled_length)
    print('\rProgress: |%s| %s%%' % (bar, percent), end = printEnd)
    if iteration == total: 
        print()


# Compute height difference between two sets of vertices
def height_difference(verts1, verts2):
    # If there is an error, just exit
    if len(verts1) != len(verts2):
        print("ERROR in height_difference: different length vertex lists")
        exit(1)

    # Perform the height difference comparison
    differences = []
    for i in range(len(verts1)):
        # One of the vertices doesn't exist yields error
        if is_zero(verts1[i]) or is_zero(verts2[i]):
            differences.append("ERROR")
            continue
        differences.append(glm.length(verts1[i] - verts2[i]))
    return differences


# Compute orientation difference between two meshes
def orientation_difference(verts1, verts2, faces):
    # If there is an error, just exit
    if len(verts1) != len(verts2):
        print("ERROR in orientation_difference: different length vertex lists")
        exit(1)

    # Construct the surface normals for each mesh
    mesh1_facenorms = []
    mesh2_facenorms = []
    for face in faces:
        if (is_zero(verts1[face[0]]) or is_zero(verts1[face[1]]) or is_zero(verts1[face[2]]) or is_zero(verts2[face[0]]) or is_zero(verts2[face[1]]) or is_zero(verts2[face[2]])):
            mesh1_facenorms.append("ERROR")
            mesh2_facenorms.append("ERROR")
            continue
        mesh1_facenorms.append(glm.normalize(glm.cross(verts1[face[1]] - verts1[face[0]], verts1[face[2]] - verts1[face[0]])))
        mesh2_facenorms.append(glm.normalize(glm.cross(verts2[face[1]] - verts2[face[0]], verts2[face[2]] - verts2[face[0]]))) 

    # Construct vertex normals for each list of vertices
    mesh1_vertexnorms = [glm.vec3(0.0, 0.0, 0.0)] * len(verts1)
    mesh2_vertexnorms = [glm.vec3(0.0, 0.0, 0.0)] * len(verts2)
    for i in range(len(faces)):
        # Ignore errors
        if (isinstance(mesh1_facenorms[i], str) or isinstance(mesh2_facenorms[i], str)):
            continue
        for j in faces[i]:
            mesh1_vertexnorms[j] = mesh1_vertexnorms[j] + mesh1_facenorms[i]
            mesh2_vertexnorms[j] = mesh2_vertexnorms[j] + mesh2_facenorms[i]

    # Normalize the vertex normals
    for i in range(len(verts1)):
        # Handle errors
        if (is_zero(mesh1_vertexnorms[i]) or is_zero(mesh2_vertexnorms[i])):
            mesh1_vertexnorms[i] = "ERROR"
            mesh2_vertexnorms[i] = "ERROR"
            continue
        mesh1_vertexnorms[i] = glm.normalize(mesh1_vertexnorms[i])
        mesh2_vertexnorms[i] = glm.normalize(mesh2_vertexnorms[i])
        
    # Perform the orientation difference comparison
    differences = []
    for i in range(len(mesh1_vertexnorms)):
        # Handle errors
        if (isinstance(mesh1_vertexnorms[i], str) or isinstance(mesh2_vertexnorms[i], str)):
            differences.append("ERROR")
            continue
        differences.append(math.degrees(math.acos(glm.dot(mesh1_vertexnorms[i], mesh2_vertexnorms[i]))))

    return differences


# Get point in triangle closest to a given point
def closest_point(vertex, triangle):
    # Extract points from triangle
    point0 = triangle[0]
    point1 = triangle[1]
    point2 = triangle[2]

    # Compute direction
    direction = glm.cross((point1 - point0), (point2 - point0))
    # Make sure direction is normalized
    direction = glm.normalize(direction)

    # Create variables for implementation of Cramer's rule
    a = point0.x - point1.x
    b = point0.y - point1.y
    c = point0.z - point1.z
    d = point0.x - point2.x
    e = point0.y - point2.y
    f = point0.z - point2.z
    g = direction.x
    h = direction.y
    i = direction.z
    j = point0.x - vertex.x
    k = point0.y - vertex.y
    l = point0.z - vertex.z

    # Create reusable variables for Cramer's rule
    ei_hf = e * i - h * f
    gf_di = g * f - d * i
    dh_eg = d * h - e * g
    ak_jb = a * k - j * b
    jc_al = j * c - a * l
    bl_kc = b * l - k * c

    # Solve values of beta and gamma
    M = a * ei_hf + b * gf_di + c * dh_eg
    beta = (j * ei_hf + k * gf_di + l * dh_eg) / M
    gamma = (i * ak_jb + h * jc_al + g * bl_kc) / M
    t = -1.0 * (f * ak_jb + e * jc_al + d * bl_kc) / M

    # Find new point in plane
    P = vertex + t * direction

    # Case 1: P is within the triangle
    if (beta >= 0 and gamma >= 0 and beta + gamma <= 1):
        return P

    # Case 2: Outside of triangle on side beta < 0
    if (beta < 0):
        proj_val = glm.dot(P - point0, point2 - point0) / glm.dot(point2 - point0, point2 - point0)
        if proj_val < 0.0:
            return point0
        if proj_val > 1.0:
            return point2
        return point0 + proj_val * (point2 - point0)

    # Case 3: Outside of triangle on side gamma < 0
    if (gamma < 0):
        proj_val = glm.dot(P - point0, point1 - point0) / glm.dot(point1 - point0, point1 - point0)
        if proj_val < 0.0:
            return point0
        if proj_val > 1.0:
            return point1
        return point0 + proj_val * (point1 - point0)

    # Case 4: Outside of triangle on side beta + gamma > 1
    proj_val = glm.dot(P - point1, point2 - point1) / glm.dot(point2 - point1, point2 - point1)
    if proj_val < 0.0:
        return point1
    if proj_val > 1.0:
        return point2
    return point1 + proj_val * (point2 - point1)


# Compute Sampled Hausdorff Distances
def sampled_hausdorff_distance(avatar_verts_proj, scan_verts_proj, scan_verts, scan_faces):
    scan_tris = gen_tris(scan_verts, scan_faces)
    shd_values = []
    completed_iterations = 0
    total_iterations = len(avatar_verts_proj)
    for i in range(len(avatar_verts_proj)):
        if (is_zero(avatar_verts_proj[i]) or is_zero(scan_verts_proj[i])):
            shd_values.append("ERROR")
        else:
            vertex = avatar_verts_proj[i]
            current_min = float('inf')
            for triangle in scan_tris:
                closest = closest_point(vertex, triangle)
                current_dist = glm.length(vertex - closest)
                if (current_dist < current_min):
                    current_min = current_dist
            shd_values.append(current_min)
        completed_iterations += 1
        if (completed_iterations % 25 == 0):
            print_progress(completed_iterations, total_iterations)
    print_progress(completed_iterations, total_iterations)
    return shd_values


# Generates a histogram of input data
def generate_histogram(data, num_bins=100, title="", xlabel="", output_name="out.png", factor=1.0):
    filtered_data = [val * factor for val in data if not isinstance(val, str)]
    bin_edges = (numpy.linspace(0.0, max(filtered_data), num_bins + 1)).tolist()
    plt.figure(figsize=(10, 8), dpi=80)
    x_bounds = [0.0, max(filtered_data)]
    plt.xlim(x_bounds)
    plt.hist(filtered_data, bins=bin_edges, weights=(numpy.ones_like(filtered_data) / float(len(filtered_data))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Proportion of data")
    plt.savefig(output_name)
    plt.close()


# Print output to console and output file
def text_out(hd_vals, od_vals, shd_vals, output_name="out.txt"):
    filtered_hd = [val for val in hd_vals if not isinstance(val, str)]
    filtered_od = [val for val in od_vals if not isinstance(val, str)]
    filtered_shd = [val for val in shd_vals if not isinstance(val, str)]

    print("========================================")
    missing_verts = len(hd_vals) - len(filtered_hd) - 3
    print("Missing vertices: {:d}".format(missing_verts))
    print("========================================")
    hd_min = numpy.min(filtered_hd)*1000
    hd_max = numpy.max(filtered_hd)*1000
    hd_mean = numpy.mean(filtered_hd)*1000
    hd_std = numpy.std(filtered_hd)*1000
    print("Height Difference (HD) statistics:")
    print("Min: {:.4f}mm".format(hd_min))
    print("Max: {:.4f}mm".format(hd_max))
    print("Mean: {:.4f}mm".format(hd_mean))
    print("SDev: {:.4f}mm".format(hd_std))
    print("========================================")
    od_min = numpy.min(filtered_od)
    od_max = numpy.max(filtered_od)
    od_mean = numpy.mean(filtered_od)
    od_std = numpy.std(filtered_od)
    print("Orientation Difference (OD) statistics:")
    print("Min: {:.4f} degrees".format(od_min))
    print("Max: {:.4f} degrees".format(od_max))
    print("Mean: {:.4f} degrees".format(od_mean))
    print("SDev: {:.4f} degrees".format(od_std))
    print("========================================")
    shd_min = numpy.min(filtered_shd)*1000
    shd_max = numpy.max(filtered_shd)*1000
    shd_mean = numpy.mean(filtered_shd)*1000
    shd_std = numpy.std(filtered_shd)*1000
    print("Sampled Hausdorff Distance (SHD) statistics:")
    print("Min: {:.4f}mm".format(shd_min))
    print("Max: {:.4f}mm".format(shd_max))
    print("Mean: {:.4f}mm".format(shd_mean))
    print("SDev: {:.4f}mm".format(shd_std))
    print("========================================")

    outfile = open(output_name, "w")
    outfile.write("========================================\n")
    outfile.write("Missing vertices: {:d}\n".format(missing_verts))
    outfile.write("========================================\n")
    outfile.write("Height Difference (HD) statistics:\n")
    outfile.write("Min: {:.4f}mm\n".format(hd_min))
    outfile.write("Max: {:.4f}mm\n".format(hd_max))
    outfile.write("Mean: {:.4f}mm\n".format(hd_mean))
    outfile.write("SDev: {:.4f}mm\n".format(hd_std))
    outfile.write("========================================\n")
    outfile.write("Orientation Difference (OD) statistics:\n")
    outfile.write("Min: {:.4f} degrees\n".format(od_min))
    outfile.write("Max: {:.4f} degrees\n".format(od_max))
    outfile.write("Mean: {:.4f} degrees\n".format(od_mean))
    outfile.write("SDev: {:.4f} degrees\n".format(od_std))
    outfile.write("========================================\n")
    outfile.write("Sampled Hausdorff Distance (SHD) statistics:\n")
    outfile.write("Min: {:.4f}mm\n".format(shd_min))
    outfile.write("Max: {:.4f}mm\n".format(shd_max))
    outfile.write("Mean: {:.4f}mm\n".format(shd_mean))
    outfile.write("SDev: {:.4f}mm\n".format(shd_std))
    outfile.write("========================================\n")
    outfile.close()
