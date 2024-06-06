import open3d as o3d

pcd = o3d.io.read_point_cloud("ply/car_4_7_7_praxis.ply")

# удаление точек с отклонениями
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Сохранение без шума
o3d.io.write_point_cloud("ply/denoised_cloud.ply", pcd)

# Визуализация
o3d.visualization.draw_geometries([pcd])

# Применение фильтрации MLS для увеличения плотности точек (что это)
# pcd = pcd.voxel_down_sample(voxel_size=0.02)

# Визуализация
# o3d.visualization.draw_geometries([pcd])

# Восстановление поверхности с использованием алгоритма Poisson
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Преобразование поверхности обратно в облако точек
pcd_dense = mesh.sample_points_poisson_disk(number_of_points=len(pcd.points) * 10)

# Визуализация
o3d.visualization.draw_geometries([pcd_dense])

## TODO Poisson
# import open3d as o3d

# # Загрузка облака точек из файла .ply
# pcd = o3d.io.read_point_cloud("last_box_praxis.ply")

# # Удаление шума
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# # Применение фильтрации MLS для увеличения плотности точек
# pcd = pcd.voxel_down_sample(voxel_size=0.02)

# # Восстановление нормалей для облака точек
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# # Восстановление поверхности с использованием алгоритма Poisson
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # Преобразование поверхности обратно в облако точек
# pcd_dense = mesh.sample_points_poisson_disk(number_of_points=len(pcd.points) * 10)

# # Визуализация
# o3d.visualization.draw_geometries([pcd_dense])
