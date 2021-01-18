"""
This class generates surfaces from GIF parcellation

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import vtk
import numpy as np

import SimpleITK as sitk

from Framework.Patient import MedicalImages

from Framework.Tools import FileSystem
from Framework.Tools import NiftyReg


class Surfaces:

    def __init__(self):
        self.filesystem = FileSystem.FileSystem()
        self.registration = NiftyReg.NiftyReg()

        self.polydata = {}

    def load(self, directory=None, image=None, gif=None):
        poly = None

        stl_folder = os.path.join(directory, 'stl')
        stl_file = os.path.join(stl_folder, str(gif)+'.stl')

        if not os.path.exists(stl_folder):
            self.filesystem.create_dir(stl_folder)
        if not os.path.exists(stl_file):
            if gif not in [83,86,91,96,97]:
                poly = self.create_stl(image=image, gif=gif, file=stl_file)
        else:
            poly = self.open_stl(file=stl_file)

        self.polydata[gif] = poly
        return poly

    def load_scalp(self, stl_folder=None, image=None):
        poly = None
        stl_file = os.path.join(stl_folder, 's_scalp.stl')

        if not os.path.exists(stl_folder):
            self.filesystem.create_dir(stl_folder)
        if not os.path.exists(stl_file):
            poly = self.create_stl_scalp(image=image, file=stl_file)
        else:
            poly = self.open_stl(file=stl_file)

        return poly

    def load_structure(self, stl_folder=None, image=None, structure=None):
        poly = None

        stl_file = os.path.join(stl_folder, 's_' + structure + '.stl')

        if not os.path.exists(stl_folder):
            self.filesystem.create_dir(stl_folder)
        if not os.path.exists(stl_file):
            if structure == 'cortex':
                poly = self.create_stl(image=image, gif=[5], file=stl_file)
            elif structure == 'white':
                poly = self.create_stl(image=image, gif=MedicalImages.wmGif, file=stl_file)
            elif structure == 'deep':
                poly = self.create_stl(image=image, gif=MedicalImages.dmGif, file=stl_file)
        else:
            poly = self.open_stl(file=stl_file)

        return poly

    def create_stl_scalp(self, image=None, file=None):
        ''' Dogdas2005. Segmentation of Skull and Scalp in 3-D Human MRI Using Mathematical Morphology '''
        # TODO
        # t_skull = self.compute_skull_threshold()

        return None

    def create_stl(self, image=None, gif=None, file=None):
        '''
        gif: three posible values
            int:        get specified region
            len([])==1: lower threshold of binary mask
            len([])>1:  get region of combined labels
        '''
        # directory
        dir_end = file.rfind('\\')
        dir = file[:dir_end]
        filename = file[dir_end + 1:]
        name_end = filename.rfind('.')
        name = filename[:name_end]

        # create mask
        mask_image = None
        if isinstance(gif, int):
            # create mask of single region
            mask_image = (image == gif)
        elif len(gif) == 1:
            # take region as lowest threshold
            binaryThresholdFilter = sitk.BinaryThresholdImageFilter()
            binaryThresholdFilter.SetLowerThreshold(gif[0])
            binaryThresholdFilter.SetUpperThreshold(208)
            mask_image = binaryThresholdFilter.Execute(image)
        elif len(gif) > 1:
            # union of masks of each region
            maskS = (image == gif[0])
            for i in range(1, len(gif)):
                maskS = maskS | (image == gif[i])

        # write NII
        # https://simpleitk.readthedocs.io/en/master/Documentation/docs/source/IO.html
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        writer.SetFileName(os.path.join(dir, name + '.nii.gz'))
        writer.Execute(mask_image)

        # read NII
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(os.path.join(dir, name + '.nii.gz'))
        reader.Update()

        # transformation matrix
        header = reader.GetNIFTIHeader()
        imgV = reader.GetOutput()
        mat = self.compute_transformation_matrix(header=header, image=imgV)

        # mask to vtk
        dim = reader.GetOutput().GetDimensions()
        centre = reader.GetOutput().GetOrigin()
        spacing = reader.GetOutput().GetSpacing()
        # print('vtk.vtkNIFTIImageReader() dim={} origin={} spacing={}'.format(dim, centre, spacing))
        # print('GIF dim={} origin={} spacing={}'.format(image.GetSize(), image.GetOrigin(), image.GetSpacing()))

        # gaussian
        gaussian = vtk.vtkImageGaussianSmooth()
        gaussian.SetInputConnection(reader.GetOutputPort())
        gaussian.SetDimensionality(3)
        gaussian.SetRadiusFactor(0.49)
        gaussian.SetStandardDeviation(0.1)
        gaussian.ReleaseDataFlagOn()
        gaussian.UpdateInformation()
        gaussian.Update()

        # marching cubes
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(gaussian.GetOutputPort())
        dmc.ComputeNormalsOn()
        dmc.SetValue(0, 1)
        dmc.Update()
        if dmc.GetOutput().GetNumberOfPoints() == 0:
            print('marching cubes of GIF={} is {}'.format(gif, dmc.GetOutput().GetNumberOfPoints()))
            return None

        # smooth marching cubes
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoothingIterations = 30  # 15 10
        passBand = 0.001  # 2
        featureAngle = 60.0  # 120.0 360.0
        smoother.SetInputConnection(dmc.GetOutputPort())
        smoother.SetNumberOfIterations(smoothingIterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()  # on
        smoother.SetFeatureAngle(featureAngle)
        smoother.SetPassBand(passBand)
        smoother.NonManifoldSmoothingOn()
        smoother.BoundarySmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        # translate
        transform = vtk.vtkPerspectiveTransform()
        transform.SetMatrix(mat)
        # transform.Concatenate(matA)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(smoother.GetOutputPort())
        transformFilter.Update()

        # transform if M was saved
        polydata = transformFilter.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            print('Number of mesh points', polydata.GetNumberOfPoints())
            return None

        # create normals if not available
        point_normals = polydata.GetPointData().GetNormals()
        if point_normals == None:
            normalGen = vtk.vtkPolyDataNormals()
            normalGen.SetInputData(polydata)
            normalGen.AutoOrientNormalsOn()

            normalGen.Update()
            point_normals = normalGen.GetOutput().GetPointData().GetNormals()
            polydata.GetPointData().SetNormals(point_normals)
            polydata.GetPointData().GetNormals().Modified()
            polydata.GetPointData().Modified()

        # save STL
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileName(file)
        # stlWriter.SetInputConnection(smoother.GetOutputPort())
        # stlWriter.SetInputConnection(transformFilter.GetOutputPort())
        stlWriter.SetInputData(polydata)
        stlWriter.Write()

        return polydata

    def open_stl(self, file=None, flip_normals=False):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file)
        reader.Update()

        polydata = reader.GetOutput()

        # create normals if not available
        point_normals = polydata.GetPointData().GetNormals()
        if point_normals == None:
            normalGen = vtk.vtkPolyDataNormals()
            normalGen.SetInputData(polydata)
            normalGen.AutoOrientNormalsOn()

            if flip_normals:
                normalGen.FlipNormalsOn()

            normalGen.Update()
            point_normals = normalGen.GetOutput().GetPointData().GetNormals()
            polydata.GetPointData().SetNormals(point_normals)
            polydata.GetPointData().GetNormals().Modified()
            polydata.GetPointData().Modified()

        print('STL: ', file)
        print('   points = ', polydata.GetPoints().GetNumberOfPoints())
        print('   arrays = ', polydata.GetPointData().GetNumberOfArrays())
        print('   arrays = ', polydata.GetPointData().GetNormals().GetSize())
        print('   tuples = ', polydata.GetPointData().GetNumberOfTuples())
        print('   tuple normals = ', polydata.GetPointData().GetNormals().GetNumberOfTuples())

        return polydata

    def compute_transformation_matrix(self, header=None, image=None):
        s = image.GetSpacing()
        s = np.array([s[0], s[1], s[2], 1])
        ori = np.array([header.GetQOffsetX(), header.GetQOffsetY(), header.GetQOffsetZ(), 1])

        '''Directions'''
        # https://fromosia.wordpress.com/2017/03/10/image-orientation-vtk-itk/
        # Use QForm matrix
        mat = vtk.vtkMatrix4x4()
        mat.Identity()
        if (header.GetQFormCode() > 0):
            b = header.GetQuaternB()
            c = header.GetQuaternC()
            d = header.GetQuaternD()
            a = np.sqrt(1 - b * b - c * c - d * d)
            A = np.array([
                [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c, ori[0]],
                [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b, ori[1]],
                [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b, ori[2]],
                [0, 0, 0, 1]
            ])
            # print('A\n', A)
            # negate x [L->R]
            A[0:3, 0] *= -1.0
            A[0, 3] *= -1.0
            # negate y [P->A]
            A[0:3, 1] *= -1.0
            A[1, 3] *= -1.0
        #
        #     # convert to RAS
        #     axcodes = nib.aff2axcodes(_gifN.affine)
        #     print('axcodes', axcodes)
        #     if axcodes[0] == 'L' and axcodes[1] == 'P' and axcodes[2] == 'S':
        #         # negate x [L->R]
        #         A[0:3, 0] *= -1.0
        #         A[0, 3] *= -1.0
        #         # negate y [P->A]
        #         A[0:3, 1] *= -1.0
        #         A[1, 3] *= -1.0
        #     if axcodes[0] == 'L' and axcodes[1] == 'I' and axcodes[2] == 'P':
        #         # negate x [L->R]
        #         A[0:3, 0] *= -1.0
        #         A[0, 3] *= -1.0
        #         # negate y [P->A]
        #         A[0:3, 1] *= -1.0
        #         A[1, 3] *= -1.0
        #         # # negate z [I->S]
        #         A[0:2, 2] *= -1.0
        #         A[2, 0:2] *= -1.0
        #         print('after negate A\n', A)
        #         # invert
        #         A = np.linalg.inv(A)
        #         # negate again
        #         A[0, 2] *= -1.0
        #         A[0, 3] *= -1.0
        #         A[1, 2] *= -1.0
        #         A[1, 3] *= -1.0
        #         A[2, 0] *= -1.0
        #         A[2, 1] *= -1.0
        #         np.savetxt(os.path.join(dir, name + '.txt'), A)
        #         A = np.eye(4, 4)
        #     if axcodes[0] == 'L' and axcodes[1] == 'S' and axcodes[2] == 'P':
        #         # negate x [L->R]
        #         A[0:3, 0] *= -1.0
        #         A[0, 3] *= -1.0
        #         # negate y [P->A]
        #         A[0:3, 1] *= -1.0
        #         A[1, 3] *= -1.0
        #         # # negate z [I->S]
        #         # A[0:2, 2] *= -1.0
        #         # A[2, 0:2] *= -1.0
        #         print('after negate A\n', A)
        #         # invert
        #         A = np.linalg.inv(A)
        #         # negate again
        #         A[0, 2] *= -1.0
        #         A[0, 3] *= -1.0
        #         A[1, 2] *= -1.0
        #         A[1, 3] *= -1.0
        #         A[2, 0] *= -1.0
        #         A[2, 1] *= -1.0
        #         np.savetxt(os.path.join(dir, name + '.txt'), A)
        #         A = np.eye(4, 4)
        #
        #     print('after A\n', A)
            # Obtain user transform for vtk algorithms
            # mat = vtk.vtkMatrix4x4()
            [[mat.SetElement(i, j, A[i, j]) for i in range(4)] for j in range(4)]
            # print("From qform: \n", mat)

        # Use SForm Matrix
        if (header.GetSFormCode() > 0):
            gx = header.GetSRowX()
            gy = header.GetSRowY()
            gz = header.GetSRowZ()
            # divide SForm matrix by spacing
            gx /= s
            gy /= s
            gz /= s
            A = np.zeros([4, 4])
            A[3, 3] = 1
            A[0, :] = gx
            A[1, :] = gy
            A[2, :] = gz
            # Obtain user transform for vtk algorithms
            # mat = vtk.vtkMatrix4x4()
            [[mat.SetElement(i, j, A[i, j]) for i in range(4)] for j in range(4)]
            # print("From SForm: \n", mat)

        return mat

    def transform_polydata(self, polydata=None, ref_image=None, M=None, stl_folder=None, filename=None):
        ''' Transform an STL given an affine matrix '''
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()
        # print('num_points=',num_points)

        stl_points = []
        for i in range(num_points):
            # get point
            pi = [0., 0., 0.]
            points.GetPoint(i, pi)
            stl_points.append(pi)
        # stl_points = np.asarray(stl_points)
        # print('stl_points', stl_points)

        in_file = os.path.join(stl_folder, filename+'_patient.txt')
        out_file = os.path.join(stl_folder, filename + '.txt')
        stl_file = os.path.join(stl_folder, filename + '.stl')
        self.filesystem.save_points_as_txt(points=stl_points, file=in_file)
        self.registration.transform_points(ref_img=ref_image, trans_matrix=M, in_points=in_file, out_points=out_file)
        self.fix_output(file=out_file)
        txt_points = self.filesystem.get_points_from_txt(file=out_file, delim=' ')
        # print('txt_points = ', len(txt_points))

        # create new polydata
        t_surface_polydata = vtk.vtkPolyData()
        t_surface_polydata.DeepCopy(polydata)

        # transform polydata
        points = t_surface_polydata.GetPoints()
        for i in range(num_points):
            # get point
            # pi = [0., 0., 0.]
            # points.GetPoint(i, pi)

            # transform
            # flo_phy = np.array([pi[0], pi[1], pi[2]])
            # p = np.array([flo_phy[0], flo_phy[1], flo_phy[2], 1.0])
            # ref_phy = M.dot(p)[0:3]

            # set point
            # points.SetPoint(i, ref_phy[0], ref_phy[1], ref_phy[2])
            p = txt_points[i]
            # print('p[{}] = [{}, {}, {}]'.format(i, p[0], p[1], p[2]))
            points.SetPoint(i, p[0], p[1], p[2])
        t_surface_polydata.SetPoints(points)

        # create normals
        normalGen = vtk.vtkPolyDataNormals()
        normalGen.SetInputData(t_surface_polydata)
        normalGen.AutoOrientNormalsOn()
        normalGen.Update()
        point_normals = normalGen.GetOutput().GetPointData().GetNormals()
        t_surface_polydata.GetPointData().SetNormals(point_normals)
        t_surface_polydata.GetPointData().GetNormals().Modified()
        t_surface_polydata.GetPointData().Modified()

        # Write the transformed stl file to disk
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileName(stl_file)
        stlWriter.SetInputData(t_surface_polydata)  # SetInputConnection
        stlWriter.Write()

        return t_surface_polydata

    def create_bstree(self, polydata=None):
        tree = vtk.vtkModifiedBSPTree()
        tree.SetDataSet(polydata)
        tree.BuildLocator()

        # print('vtkModifiedBSPTree = \n', tree)
        return tree

    def create_kdtree(self, polydata=None):
        tree = vtk.vtkKdTreePointLocator()
        tree.SetDataSet(polydata)
        tree.BuildLocator()

        return tree

    def fix_output(self, file=None):
        ''' An empty line is always added at the end of the file, and this causes problems when calling reg_transform again'''
        if os.path.exists(file):
            rfile = open(file)
            lines = rfile.readlines()
            rfile.close()
            # print('Lines')
            # for i in range(len(lines)):
            #     print('line={} empty={} val={}'.format(i, lines[i] == '', lines[i][:-1]))
            wfile = open(file, 'w')
            for i in range(len(lines)):
                split = lines[i].split()
                wfile.write('{:.4f} {:.4f} {:.4f}'.format(float(split[0]), float(split[1]), float(split[2])))
                if i < len(lines)-1:
                    wfile.write('\n')
            # wfile.writelines([item for item in lines[:-1]])
            # wfile.write(lines[-1][:-1])
            wfile.close()