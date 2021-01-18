"""
This class generates collision information of electrode trajectories

Steps:


Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import math
import numpy as np
import vtk

from Framework.Patient import Surfaces

from Framework.Tools import XMLParser


class SurfaceCollisionFilter:

    def __init__(self, images=None, electrodes=None, space=None):
        self.medicalImages = images
        self.electrodes = electrodes
        self.space = space
        self.directory = self.medicalImages.get_directory(space=space)
        self.stl_folder = os.path.join(self.directory, 'stl')

        self.xmlparser = XMLParser.XMLParser()

        self.surfaces = Surfaces.Surfaces()

    def execute(self):
        # load polydata
        self.polydata = {'scalp': self.surfaces.load_scalp(stl_folder=self.stl_folder, image=self.medicalImages.mri_img[self.space]),
                         'cortex': self.surfaces.load_structure(stl_folder=self.stl_folder, image=self.medicalImages.gif_img[self.space], structure='cortex'),
                         'white': self.surfaces.load_structure(stl_folder=self.stl_folder, image=self.medicalImages.gif_img[self.space], structure='white'),
                         'deep': self.surfaces.load_structure(stl_folder=self.stl_folder, image=self.medicalImages.gif_img[self.space], structure='deep')}

        # compute bs trees
        self.bstrees = {'scalp': self.surfaces.create_bstree(self.polydata['scalp']),
                        'cortex': self.surfaces.create_bstree(self.polydata['cortex']),
                        'white': self.surfaces.create_bstree(self.polydata['white']),
                        'deep': self.surfaces.create_bstree(self.polydata['deep'])}

        # compute kd trees
        self.kdtrees = {'scalp': self.surfaces.create_kdtree(self.polydata['scalp']),
                        'cortex': self.surfaces.create_kdtree(self.polydata['cortex']),
                        'white': self.surfaces.create_kdtree(self.polydata['white']),
                        'deep': self.surfaces.create_kdtree(self.polydata['deep'])}

        self.collision = self.compute_collision(impl=self.electrodes.impl[self.space])

    def load_polydata(self):
        # load polydata
        self.polydata = {
            'scalp': self.surfaces.load_scalp(stl_folder=self.stl_folder, image=self.medicalImages.mri_img[self.space]),
            'cortex': self.surfaces.load_structure(stl_folder=self.stl_folder,
                                                   image=self.medicalImages.gif_img[self.space], structure='cortex'),
            'white': self.surfaces.load_structure(stl_folder=self.stl_folder,
                                                  image=self.medicalImages.gif_img[self.space], structure='white'),
            'deep': self.surfaces.load_structure(stl_folder=self.stl_folder,
                                                 image=self.medicalImages.gif_img[self.space], structure='deep')}

        # compute bs trees
        self.bstrees = {'scalp': self.surfaces.create_bstree(self.polydata['scalp']),
                        'cortex': self.surfaces.create_bstree(self.polydata['cortex']),
                        'white': self.surfaces.create_bstree(self.polydata['white']),
                        'deep': self.surfaces.create_bstree(self.polydata['deep'])}

        # compute kd trees
        self.kdtrees = {'scalp': self.surfaces.create_kdtree(self.polydata['scalp']),
                        'cortex': self.surfaces.create_kdtree(self.polydata['cortex']),
                        'white': self.surfaces.create_kdtree(self.polydata['white']),
                        'deep': self.surfaces.create_kdtree(self.polydata['deep'])}

    def execute_infer(self, impl=None):
        self.collision = self.compute_collision(impl=impl)

    def compute_collision(self, impl=None):
        collision = {'name': [], 'acpcih': [],
                     'scalp': [], 'cortex': [], 'white': [], 'deep': []}

        # acpc points
        acpc_points = []
        acpc_file = os.path.join(self.medicalImages.get_directory(space=self.space), 'acpcih.mps')
        acpc_points = self.xmlparser.open_mps(file=acpc_file)

        # iterate through electrodes
        # impl = self.electrodes.impl[self.space]
        for i in range(len(impl['name'])):
            name = impl['name'][i]
            ep = impl['ep'][i]
            points = impl['points'][i]
            num_points = len(points)

            # collision with scalp
            x_from = ep
            x_to = points[num_points - 1]
            # print('     collision: ep={} x_to={}'.format(ep, x_to))
            u = x_to - x_from
            u /= np.linalg.norm(u)
            scalp_cx = self.compute_collision_point(tree=self.bstrees['scalp'], x_from=x_from, x_to=x_to)
            if scalp_cx.size == 0:
                # extend EP
                # print('{} extend EP from {} to {}'.format(name, x_from, x_from - u * 10.0))
                x_from = x_from - u * 10.0
                scalp_cx = self.compute_collision_point(tree=self.bstrees['scalp'], x_from=x_from, x_to=x_to)
            # print('     collision: x_from={} scalp_cx={}'.format(x_from, scalp_cx))
            scalp_cn = self.compute_collision_normal(tree=self.kdtrees['scalp'], polydata=self.polydata['scalp'], collision_point=scalp_cx)
            scalp_theta = self.compute_angle(a=u, b=scalp_cn)
            # print('scalp collision[{}]: from={} to={} point={} normal={} angle={} '.format(name, x_from, x_to, scalp_cx, scalp_cn, scalp_theta))

            # collision with cortex (only once)
            cortex_c = []
            cortex_cx = self.compute_collision_point(tree=self.bstrees['cortex'], x_from=x_from, x_to=x_to)
            if cortex_cx.size > 0:
                cortex_cn = self.compute_collision_normal(tree=self.kdtrees['cortex'], polydata=self.polydata['cortex'], collision_point=cortex_cx)
                cortex_theta = self.compute_angle(a=u, b=cortex_cn)
                cortex_c = [[len(points), len(points - 1), cortex_cx, cortex_cn, cortex_theta]]
                # print('cortex collision (from EP): point={} normal={} angle={} '.format(cortex_cx, cortex_cn, cortex_theta))
            else:
                cortex_c = self.compute_collision_trajectory(bstree=self.bstrees['cortex'], kdtree=self.kdtrees['cortex'], polydata=self.polydata['cortex'], points=points, ep=ep)
                if len(cortex_c) != 0:
                    cortex_c = [cortex_c[0]]
                # print('cortex collision only first one: ', cortex_c)

            # compute collision with white
            white_c = self.compute_collision_trajectory(bstree=self.bstrees['white'], kdtree=self.kdtrees['white'], polydata=self.polydata['white'], points=points, ep=ep)

            # compute collision with cortex
            deep_c = self.compute_collision_trajectory(bstree=self.bstrees['deep'], kdtree=self.kdtrees['deep'], polydata=self.polydata['deep'], points=points, ep=ep)

            # save
            collision['name'].append(name)
            collision['acpcih'].append(acpc_points)
            collision['scalp'].append([scalp_cx, scalp_cn, scalp_theta])
            collision['cortex'].append(cortex_c)
            collision['white'].append(white_c)
            collision['deep'].append(deep_c)

        return collision

    def compute_collision_point(self, tree=None, x_from=None, x_to=None):
        """ Function to compute collision point between two points """
        point = np.array([0., 0., 0.], dtype=np.float64)

        tolerance = .001
        t = vtk.mutable(0.0)  # parametric coordinate of intersection [0,1] = [p1,p2]
        x = [0.0, 0.0, 0.0]  # collision point

        # Note: for a typical use case (ray-triangle intersection), pcoords and subId will not be used
        pcoords = [0.0, 0.0, 0.0]
        subId = vtk.mutable(0)

        iD = tree.IntersectWithLine(x_from, x_to, tolerance, t, x, pcoords, subId)
        if iD != 0:
            for i in range(3):
                point[i] = x[i]
        else:
            point = np.array([], dtype=np.float64)

        return point

    def compute_collision_normal(self, tree=None, polydata=None, collision_point=None):
        """ Compute collision normal given a collision point """
        normal = np.array([0., 0., 0.], dtype=np.float64)
        n = [0.0, 0.0, 0.0]

        iD = tree.FindClosestPoint(collision_point)

        surface_normals = polydata.GetPointData().GetNormals()
        if iD < surface_normals.GetSize():
            surface_normals.GetTuple(iD, n)
            n /= np.linalg.norm(n)
            for i in range(3):
                normal[i] = n[i]

        return normal

    def compute_angle(self, a=None, b=None):
        # compute angle between two unit vectors
        thetaRad = math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
        thetaDeg = thetaRad * 180.0 / math.pi
        return thetaDeg

    def compute_angle_rad(self, a=None, b=None):
        # compute angle between two unit vectors
        thetaRad = math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
        return thetaRad

    def compute_collision_trajectory(self, bstree=None, kdtree=None, polydata=None, points=None, ep=None):
        """ Computes the collision with a mesh along a trajectory given by a set of points """
        n = len(points)
        collision = []

        # start from Ep to most proximal point (cn)
        x_from = ep
        x_to = points[n - 1]
        u = x_to - x_from
        u /= np.linalg.norm(u)
        point = self.compute_collision_point(tree=bstree, x_from=x_from, x_to=x_to)
        if point.shape[0] != 0:
            # compute normal
            normal = self.compute_collision_normal(tree=kdtree, polydata=polydata, collision_point=point)

            # compute angle between electrode trajectory and normal
            thetaDeg = self.compute_angle(a=u, b=normal)

            # add collision
            collision.append([n, n - 1, point, normal, thetaDeg])

        # iterate through points from most proximal (cn) to most distal (c0)
        for i in range(n - 1, 0, -1):
            x_from = points[i]
            x_to = points[i - 1]
            u = x_to - x_from
            u /= np.linalg.norm(u)

            point = self.compute_collision_point(tree=bstree, x_from=x_from, x_to=x_to)
            if point.shape[0] != 0:
                # compute normal
                normal = self.compute_collision_normal(tree=kdtree, polydata=polydata, collision_point=point)

                # compute angle between electrode trajectory and normal
                thetaDeg = self.compute_angle(a=u, b=normal)

                # add collision
                collision.append([i, i - 1, point, normal, thetaDeg])

        # return collision_points, collision_normals
        return collision

    def compute_distance_to_triangle(self, p, p1, p2, p3):
        # http://paulbourke.net/geometry/pointlineplane/
        # compute triangle normal
        n = self.compute_triangle_normal(p1, p2, p3)

        # # equation of a plane: Ax + By + Cz + D = 0
        # # compute D using first point of triangle with its normal
        # D = -(n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2])
        #
        # # compute the minimum distance of point to the plane
        # # (Axa + Bya + Cza + D) / sqrt(A2 + B2 + C2)
        # minD = (n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + D) / np.linalg.norm(n)
        minD = self.compute_distance_to_plane(p, p1, n)

        # sign shows whether point is above/below plane
        return minD

    def compute_triangle_normal(self, p1, p2, p3):
        u = p2 - p1
        v = p3 - p1
        n = np.array([0., 0., 0.])

        n[0] = u[1] * v[2] - u[2] * v[1]
        n[1] = u[2] * v[0] - u[0] * v[2]
        n[2] = u[0] * v[1] - u[1] * v[0]

        return n

    def compute_distance_to_plane(self, x, p_x, p_n):
        # http://paulbourke.net/geometry/pointlineplane/
        # equation of a plane: Ax + By + Cz + D = 0
        # compute D using first point of triangle with its normal
        D = -(p_n[0] * p_x[0] + p_n[1] * p_x[1] + p_n[2] * p_x[2])

        # compute the minimum distance of point to the plane
        # (Axa + Bya + Cza + D) / sqrt(A2 + B2 + C2)
        minD = (p_n[0] * x[0] + p_n[1] * x[1] + p_n[2] * x[2] + D) / np.linalg.norm(p_n)

        # sign shows whether point is above/below plane
        return minD

    def compute_distance_to_line(self, p1, p2, p3):
        # Compute lateral deviation from line (p1,p2) to a point (p3)
        # http://paulbourke.net/geometry/pointlineplane/
        d = 0.0

        # compute u (interpolation of line equation)
        u = ((p3[0] - p1[0]) * (p2[0] - p1[0]) + (p3[1] - p1[1]) * (p2[1] - p1[1]) + (p3[2] - p1[2]) * (
                    p2[2] - p1[2])) / \
            (np.linalg.norm(p2 - p1) * np.linalg.norm(p2 - p1))

        # compute P (along trajectory)
        p = p1 + u * (p2 - p1)

        # compute lateral deviation
        d = np.linalg.norm(p3 - p)

        return d