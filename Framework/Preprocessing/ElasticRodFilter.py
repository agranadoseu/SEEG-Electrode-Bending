"""
This class models electrodes as elastic rods to compute bending

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import numpy as np

from Framework.Tools import FileSystem
from Framework.Tools import XMLParser





class ElasticRodFilter:

    def __init__(self, images=None, electrodes=None):
        self.medicalImages = images
        self.electrodes = electrodes

        self.filesystem = FileSystem.FileSystem()
        self.xmlparser = XMLParser.XMLParser()

        # acpc points
        patient_acpc_points = self.load_acpc_coord(space='patient')
        acpc_points = self.load_acpc_coord(space='acpc')
        mni_acpc_points = self.load_acpc_coord(space='mni_aff')

        # material frames
        self.frame = {'patient': self.compute_material_frame(points=patient_acpc_points),
                      'acpc': self.compute_material_frame(points=acpc_points),
                      'mni_aff': self.compute_material_frame(points=mni_acpc_points)}

    def execute(self):
        # iterate through spaces
        for s in self.frame.keys():
            directory = self.electrodes.get_directory(type='impl', space=s, suffix='i')
            # bending_dir = os.path.join(directory, 'bending')
            # if not os.path.exists(bending_dir):
            #     self.filesystem.create_dir(bending_dir)

            data = self.electrodes.impl[s]

            # iterate through electrodes
            for e in range(len(data['name'])):
                name = data['name'][e]
                id = data['id'][e]
                num_contacts = data['num_contacts'][e]
                ep = data['ep'][e]
                points = data['points'][e]
                # print('name = {}, num_points = {}'.format(name, len(points)))

                # filename
                filename = os.path.join(directory, 'G'+name[1:]+'.xmlE')

                # if file does not exist
                if not os.path.exists(filename):

                    # compute ghost points
                    gEp, mEp, gPoints, gMids = self.compute_ghost_points(ep=ep, points=points, frame=self.frame[s])

                    # save ghost points
                    self.xmlparser.save_points_as_xml(name='G'+name[1:], ep=gEp, points=gPoints, colour=np.array([.8, 1., .8]), xml_file=filename)


    def compute_material_frame(self, points=None):
        # p0, p1 are contact points; p2 is a ghost point
        [p0, p1, p2] = points

        frame = np.zeros((3, 3), dtype=np.float64)

        # compute directions
        d3 = p1 - p0
        d3 /= np.linalg.norm(d3)
        d2 = np.cross(d3, p2 - p0)
        d2 /= np.linalg.norm(d2)
        d1 = np.cross(d2, d3)

        # fill matrix
        for i in range(3):
            frame[i, 0] = d2[i]  # x
            frame[i, 1] = d1[i]  # y
            frame[i, 2] = d3[i]  # z

        return frame

    def load_acpc_coord(self, space=None):
        acpc_points = []
        acpc_file = os.path.join(self.medicalImages.get_directory(space=space), 'acpcih.mps')
        if os.path.exists(acpc_file):
            acpc_points = self.xmlparser.open_mps(file=acpc_file)
            if len(acpc_points) >= 3:
                # get points
                ac, pc, ih = acpc_points[0], acpc_points[1], acpc_points[2]
                #print('ac={}, pc={}, ih={}'.format(ac, pc, ih))
        else:
            print('ERROR: acpc file = {} do not exist'.format(acpc_file))

        return acpc_points

    def compute_ghost_points(self, ep=None, points=None, frame=None):
        # create array
        gPoints = np.zeros((len(points) - 1, 3), dtype=np.float64)
        gMids = np.zeros((len(points) - 1, 3), dtype=np.float64)

        # same distance from trajectory to ghost points
        d = np.linalg.norm(points[0] - points[1])

        # iterate through pairs of contacts (points go from TP to EP)
        for i in range(len(gPoints)):
            # always towards TP
            ci = points[i + 1]
            cj = points[i]

            mid = self.compute_mid_point(p1=ci, p2=cj)
            v = self.compute_perpendicular_line(p1=ci, p2=cj, m=mid, ref=frame[:, 1])
            gPoints[i] = mid + 0.5 * d * v
            gMids[i] = mid

        # ghost entry point
        gMid = self.compute_mid_point(p1=ep, p2=points[-1])
        v = self.compute_perpendicular_line(p1=ep, p2=points[-1], ref=frame[:, 1])
        gEp = gMid + 0.5 * d * v

        return gEp, gMid, gPoints, gMids

    def compute_mid_point(self, p1=None, p2=None):
        ''' Compute mid point between two points '''
        # direction of line
        u = p2 - p1

        # point half way
        t = 0.5
        mid_point = p1 + t * u

        return mid_point

    def compute_perpendicular_line(self, p1=None, p2=None, m=None, ref=None):
        ''' Compute perpendicular line between two points given a reference direction '''
        # Compute perpendicular line given a reference (frame)
        d3 = p2 - p1
        d3 /= np.linalg.norm(d3)
        d1 = np.cross(d3, ref)
        d1 /= np.linalg.norm(d1)
        d2 = np.cross(d1, d3)
        d2 /= np.linalg.norm(d2)

        return d2

    def compute_omega(self, fi=None, fj=None):
        # compute Darvoux vector
        omega = np.array([0., 0., 0.], dtype=np.float64)

        # in PBD midEdgeLength is constant and equal to 1
        # However, it may be the case that the closer the distance between particles, the larger the factor (dividend)
        midEdgeLength = 1.0

        # orthogonal components of frames are column vectors
        # _fa[:,0]

        # When comparing two frames, factor measures how equal each component is (if parallel the largest)
        # values range from -1 (parallel opposing - 180deg), through 0 (perpendicular) to +1 (parallel with same direction - 0deg)
        # if frames are equal (factor=0.5) and if completely opposing (factor=-1)
        factor = 1.0 + np.dot(fi[:, 0], fj[:, 0]) + np.dot(fi[:, 1], fj[:, 1]) + np.dot(fi[:, 2], fj[:, 2])
        factor = 2.0 / (midEdgeLength * factor)

        # permutation
        permutation = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int16)
        for p in range(3):
            i = permutation[p][0]
            j = permutation[p][1]
            k = permutation[p][2]
            omega[p] = np.dot(fi[:, j], fj[:, k]) - np.dot(fi[:, k], fj[:, j])
            omega[p] *= factor

        return omega

    def compute_curvature(self, points):
        # https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        # input: 3D points
        # output: velocity (nx3), speed (nx1), tangent (nx3), normal (nx3), acceleration (nx3), k (nx1)
        # print("\nData")
        # print(data.shape)

        # compute velocity gradients
        dx_dt = np.gradient(points[:, 0])
        dy_dt = np.gradient(points[:, 1])
        dz_dt = np.gradient(points[:, 2])
        # dx_dt = np.gradient(data[:, 0]) * -1.0
        # dy_dt = np.gradient(data[:, 1]) * -1.0
        # dz_dt = np.gradient(data[:, 2]) * -1.0
        velocity = np.array([[dx_dt[i], dy_dt[i], dz_dt[i]] for i in range(dx_dt.size)])
        # print("\nVelocity")
        # print(velocity)
        # print(velocity.shape)

        # compute speed
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt)
        # print("\nSpeed")
        # print(ds_dt)

        # compute tangent which has the same direction as velocity but is a unit vector
        tangent = np.array([1 / ds_dt] * 3).transpose() * velocity
        # print("\nTangent")
        # print(tangent)

        # compute normal which is the derivative of the tangent
        dT_x = np.gradient(tangent[:, 0])
        dT_y = np.gradient(tangent[:, 1])
        dT_z = np.gradient(tangent[:, 2])
        dT_dt = np.array([[dT_x[i], dT_y[i], dT_z[i]] for i in range(dT_x.size)])
        dT_dt_length = np.sqrt(dT_x * dT_x + dT_y * dT_y + dT_z * dT_z)
        for i in np.where(dT_dt_length == 0.0)[0]:
            dT_dt_length[i] = 0.00001   # avoid division by zero
        normal = np.array([1 / dT_dt_length] * 3).transpose() * dT_dt
        # print("\nNormal")
        # print(normal)

        # compute second derivatives of s, x, y wrt t
        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        d2z_dt2 = np.gradient(dz_dt)

        # compute curvature of curves in 3D space
        k = np.sqrt(
            np.square(d2z_dt2 * dy_dt - d2y_dt2 * dz_dt) + np.square(d2x_dt2 * dz_dt - d2z_dt2 * dx_dt) + np.square(
                d2y_dt2 * dx_dt - d2x_dt2 * dy_dt)) / (dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt) ** 1.5
        # print("\nCurvature")
        # print(k)
        # print(k.shape)

        # compute acceleration
        t_component = np.array([d2s_dt2] * 3).transpose()
        n_component = np.array([k * ds_dt * ds_dt] * 3).transpose()
        acceleration = t_component * tangent + n_component * normal
        da_dt = np.sqrt(acceleration[:,0] * acceleration[:,0] + acceleration[:,1] * acceleration[:,1] + acceleration[:,2] * acceleration[:,2])

        # # plot
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.plot(data[:, 0], data[:, 1], data[:, 2], label='points', lw=2, c='Dodgerblue')
        # ax.quiver(data[:, 0], data[:, 1], data[:, 2], velocity[:, 0], velocity[:, 1], velocity[:, 2], length=1, normalize=False, colors='black')
        # ax.quiver(data[:, 0], data[:, 1], data[:, 2], tangent[:, 0], tangent[:, 1], tangent[:, 2], length=1, normalize=False, colors='red')
        # ax.quiver(data[:, 0], data[:, 1], data[:, 2], normal[:, 0], normal[:, 1], normal[:, 2], length=1, normalize=False, colors='green')
        # # ax.set_xlim(0, 10)
        # # ax.set_ylim(0, 10)
        # # ax.set_zlim(0, 10)
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # plt.show()

        return velocity, ds_dt, tangent, normal, acceleration, da_dt, k
