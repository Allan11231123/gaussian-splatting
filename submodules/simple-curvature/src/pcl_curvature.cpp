#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_cloud.h>

namespace py = pybind11;

// Compute both 'Normal' and 'Curvature'
py::array_t<double> compute_curvature_and_normal(py::array_t<double> input_array, double search_radius) {
    // 1. Check input data
    py::buffer_info buf = input_array.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input must be a shape of (N, 3)");
    }

    size_t num_points = buf.shape[0];
    auto ptr = static_cast<double *>(buf.ptr);
    
    // safe check A
    if (num_points == 0) {
        // return py::array_t<double>({0, 8});
        throw std::runtime_error("Input point cloud is empty.");
    }

    // 2. Convert: NumPy -> pcl::PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->width = num_points;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < num_points; ++i) {
        cloud->points[i].x = ptr[i * 3 + 0];
        cloud->points[i].y = ptr[i * 3 + 1];
        cloud->points[i].z = ptr[i * 3 + 2];
    }

    // 3. PCL computation process
    
    // A. Build KdTree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    // B. Compute Normal
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(search_radius);
    ne.compute(*normals);

    // safe check B
    if (normals->points.size() != num_points) {
        throw std::runtime_error("Normal estimation failed: output size mismatch.");
    }

    // C. Compute Principal Curvatures
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> pce;
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());

    pce.setInputCloud(cloud);
    pce.setInputNormals(normals);
    pce.setSearchMethod(tree);
    pce.setRadiusSearch(search_radius);
    pce.compute(*principal_curvatures);

    // safe check C
    if (principal_curvatures->points.size() != num_points) {
        throw std::runtime_error("Principal curvature estimation failed: output size mismatch.");
    }

    // 4. Combine data for return: (N, 8)
    // [nx, ny, nz, k1, k2, pc1_x, pc1_y, pc1_z]
    
    py::array_t<double> result = py::array_t<double>({(long)num_points, (long)8});
    auto result_buf = result.request();
    double *result_ptr = static_cast<double *>(result_buf.ptr);

    for (size_t i = 0; i < num_points; ++i) {
        // --- Part 1: Normals ---
        result_ptr[i * 8 + 0] = normals->points[i].normal_x;
        result_ptr[i * 8 + 1] = normals->points[i].normal_y;
        result_ptr[i * 8 + 2] = normals->points[i].normal_z;

        // --- Part 2: Curvature Magnitudes ---
        // pc1 = Maximum Curvature
        // pc2 = Minimum Curvature
        result_ptr[i * 8 + 3] = principal_curvatures->points[i].pc1; 
        result_ptr[i * 8 + 4] = principal_curvatures->points[i].pc2; 

        // --- Part 3: Principal Direction ---
        // This is the direction vector corresponding to the maximum curvature (pc1) at the point
        result_ptr[i * 8 + 5] = principal_curvatures->points[i].principal_curvature_x;
        result_ptr[i * 8 + 6] = principal_curvatures->points[i].principal_curvature_y;
        result_ptr[i * 8 + 7] = principal_curvatures->points[i].principal_curvature_z;
    }

    return result;
}

PYBIND11_MODULE(pcl_curvature, m) {
    m.doc() = "PCL Principal Curvature & Normal Calculation Module";
    m.def("compute", &compute_curvature_and_normal, "Returns (N, 8) array: [nx, ny, nz, k1, k2, vx, vy, vz]");
}