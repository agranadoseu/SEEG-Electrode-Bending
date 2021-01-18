"""
This class uses VTK to render electrode trajectories

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import numpy as np

import vtk
from vtk.util import numpy_support

class VTKRenderer:

    def __init__(self):
        self.actors = []
        self.ids = []

    def key_pressed_callback(self, obj, event):
        key = obj.GetKeySym()
        if key == "1":
            print('Exit VTK')
            #self.close()

    def create_vtk_renderer(self):
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetFullScreen(True)
        self.renWin.AddRenderer(self.ren)

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.iren.AddObserver("KeyPressEvent", self.key_pressed_callback)

    def execute(self):
        # render list of actors
        for i in range(len(self.actors)):
            self.ren.AddActor(self.actors[i])

        self.ren.SetBackground(0.1, 0.1, 0.1)

        self.iren.Initialize()

        # axes
        axes = vtk.vtkAxesActor()
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.iren)
        widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        widget.SetEnabled(1)
        widget.InteractiveOn()
        self.ren.ResetCamera()

        self.renWin.Render()
        self.iren.Start()

    def close(self):
        self.renWin.Finalize()
        self.iren.TerminateApp()
        del self.renWin
        del self.iren

    def render_prediction(self, data=None, filter=None, colour=None, radius=1.0, opacity=0.0):
        # number of electrodes
        N = len(data['name'])
        for i in range(N):
            # filter
            if filter is not None:
                if data['id'][i] not in filter:
                    continue
                print('     filtered electrode: ', data['name'][i])

            if len(data['points'][i]) == 0:
                continue

            # predicted trajectory
            self.create_trajectory_actor(idx=data['id'][i], num_elec=N, x_points=data['points'][i], ep=data['ep'][i], colour=colour, radius=radius, opacity=opacity)

            # predicted uncertainty
            if 'lux' in list(data.keys()):
                self.create_uncertainty_actor(idx=data['id'][i], num_elec=N, x_points=data['points'][i], delta=data['points'][i], colour=colour, opacity=opacity)

    def render_trajectories(self, data=None, filter=None, colour=None, radius=1.0, opacity=0.0):
        # number of electrodes
        N = len(data['name'])
        for i in range(N):
            # filter
            if filter is not None:
                if data['id'][i] not in filter:
                    continue
                print('     filtered electrode: ', data['name'][i])

            if len(data['points'][i]) == 0:
                continue

            # trajectory
            self.create_trajectory_actor(idx=data['id'][i], num_elec=N, x_points=data['points'][i], ep=data['ep'][i], colour=colour, radius=radius, opacity=opacity)

            # stylet
            if data['name'][i].startswith('R') and data['stylet'][i] != -1:
                stylet_points = data['points'][i][int(np.round(data['stylet'][i])):]
                self.create_trajectory_actor(idx=data['id'][i], num_elec=N, x_points=stylet_points, ep=data['ep'][i], colour=colour, radius=radius/2., opacity=1.)

            # contacts
            self.create_contact_actors(idx=data['id'][i], num_elec=N, c_points=data['contacts'][i], colour=colour, radius=radius, opacity=opacity)

            # entry point
            direction = data['points'][i][-1] - data['ep'][i]
            direction /= np.linalg.norm(direction)
            self.create_ep_actors(idx=data['id'][i], num_elec=N, ep=data['ep'][i], direction=direction, colour=colour, radius=radius, opacity=opacity)

    def render_vectorfield(self, data=None, source=None, vector=None, filter=None, colour=None):
        # @TODO render also image-based vector field: https://vtk.org/Wiki/VTK/Examples/Cxx/Visualization/VectorField
        # number of electrodes
        N = len(data['name'])
        for i in range(N):
            # filter
            if filter is not None:
                if data['id'][i] not in filter:
                    continue

            if len(data[vector][i]) == 0:
                continue

            # vector field
            self.create_vectorfield_actor(idx=data['id'][i], num_elec=N, source=data[source][i], field=data[vector][i], colour=colour)

    def render_surface(self, polydata=None, colour=None, opacity=None):
        if polydata is None:
            return

        # create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colour[0], colour[1], colour[2])
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetRepresentationToWireframe()
        self.actors.append(actor)
        self.ids.append(0)

    def create_uncertainty_actor(self, idx=0, num_elec=0, x_points=None, delta=None, colour=None, opacity=0.0):
        # create cylinders between contiguous points
        lux = 0.1
        luy = 0.1
        # for p in range(len(x_points)-1):
        for p in range(len(x_points) - 1, 0, -1):
            x1 = x_points[p]
            x2 = x_points[p-1]
            lux += 0.5
            luy += 0.1

            cylinderActor = self.create_oriented_cylinder_actor(x1=x1, x2=x2, lux=lux, luy=luy, colour=colour, opacity=opacity)

            self.actors.append(cylinderActor)
            self.ids.append(idx)

    def create_trajectory_actor(self, idx=0, num_elec=0, x_points=None, ep=None, colour=None, radius=1.0, opacity=0.0):
        # points
        points = vtk.vtkPoints()
        for p in range(len(x_points)):
            x = x_points[p]
            points.InsertPoint(p, x[0], x[1], x[2])
        points.InsertPoint(len(x_points), ep[0], ep[1], ep[2])

        # spline
        spline = vtk.vtkParametricSpline()
        spline.SetPoints(points)

        # function source
        functionSource = vtk.vtkParametricFunctionSource()
        functionSource.SetParametricFunction(spline)
        functionSource.SetUResolution(10 * points.GetNumberOfPoints())
        functionSource.Update()
        n = functionSource.GetOutput().GetNumberOfPoints()

        # interpolate the scalars
        interpolatedRadius = vtk.vtkTupleInterpolator()
        interpolatedRadius.SetInterpolationTypeToLinear()
        interpolatedRadius.SetNumberOfComponents(1)

        # ORIGINAL: Generate the radius scalars
        tubeRadius = vtk.vtkDoubleArray()
        tubeRadius.SetName("TubeRadius")
        tubeRadius.SetNumberOfTuples(n)
        tMin = interpolatedRadius.GetMinimumT()
        tMax = interpolatedRadius.GetMaximumT()
        for i in range(n):
            t = (tMax - tMin) / (n - 1) * i + tMin
            r = 0.1
            # interpolatedRadius.InterpolateTuple(t, r)
            tubeRadius.SetTuple1(i, idx)

        # Add the scalars to the polydata
        tubePolyData = functionSource.GetOutput()
        if not len(colour):
            tubePolyData.GetPointData().AddArray(tubeRadius)
            tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

        # create the tubes
        tuber = vtk.vtkTubeFilter()
        tuber.SetInputData(tubePolyData)
        tuber.SetNumberOfSides(20)
        tuber.SetRadius(radius)
        # tuber.SetVaryRadiusToVaryRadiusByVector()

        # Setup mappers
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(tubePolyData)
        # lineMapper.SetScalarRange(tubePolyData.GetScalarRange())
        if not len(colour):
            lineMapper.SetScalarRange(0, num_elec)

        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tuber.GetOutputPort())
        # tubeMapper.SetScalarRange(tubePolyData.GetScalarRange())
        if not len(colour):
            tubeMapper.SetScalarRange(0, num_elec)

        # setup actors
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)

        tubeActor = vtk.vtkActor()
        tubeActor.SetMapper(tubeMapper)
        if len(colour):
            tubeActor.GetProperty().SetColor(colour[0], colour[1], colour[2])
        tubeActor.GetProperty().SetOpacity(opacity)
        # print('colour = ', tubeActor.GetProperty().GetColor())

        self.actors.append(lineActor)
        self.actors.append(tubeActor)
        self.ids.append(idx)
        self.ids.append(idx)

    def create_contact_actors(self, idx=0, num_elec=0, c_points=None, colour=None, radius=1.0, opacity=0.0):
        # contacts
        for p in range(len(c_points)):
            c = c_points[p]

            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(21)
            sphere.SetThetaResolution(21)
            sphere.SetRadius(radius * 1.5)
            sphere.SetCenter(c[0], c[1], c[2])
            sphere.Update()

            sphereMapper = vtk.vtkPolyDataMapper()

            # colours
            if not len(colour):
                num_points = sphere.GetOutput().GetPoints().GetNumberOfPoints()
                sphereScalars = vtk.vtkDoubleArray()
                sphereScalars.SetNumberOfValues(num_points)
                for i in range(num_points):
                    sphereScalars.SetValue(i, idx)
                spherePolyData = vtk.vtkPolyData()
                spherePolyData = sphere.GetOutput()
                spherePolyData.GetPointData().SetScalars(sphereScalars)
                sphereMapper.SetInputData(spherePolyData)
                sphereMapper.SetScalarRange(0, num_elec)
            else:
                sphereMapper.SetInputConnection(sphere.GetOutputPort())

            sphereActor = vtk.vtkActor()
            sphereActor.SetMapper(sphereMapper)
            if len(colour):
                sphereActor.GetProperty().SetColor(colour[0], colour[1], colour[2])
            sphereActor.GetProperty().SetOpacity(opacity)
            # sphereActor.SetBackfaceProperty(backProperty)

            self.actors.append(sphereActor)
            self.ids.append(idx)

    def create_ep_actors(self, idx=0, num_elec=0, ep=None, direction=None, colour=None, radius=1.0, opacity=0.0):
        cone = vtk.vtkConeSource()
        cone.SetResolution(51)
        cone.SetHeight(radius*5.0)
        cone.SetRadius(radius*2.0)
        cone.SetCenter(ep[0], ep[1], ep[2])
        cone.SetDirection(direction[0], direction[1], direction[2])
        cone.Update()

        coneMapper = vtk.vtkPolyDataMapper()

        # colours
        if not len(colour):
            num_points = cone.GetOutput().GetPoints().GetNumberOfPoints()
            coneScalars = vtk.vtkDoubleArray()
            coneScalars.SetNumberOfValues(num_points)
            for i in range(num_points):
                coneScalars.SetValue(i, idx)
            conePolyData = vtk.vtkPolyData()
            conePolyData = cone.GetOutput()
            conePolyData.GetPointData().SetScalars(coneScalars)
            coneMapper.SetInputData(conePolyData)
            coneMapper.SetScalarRange(0, num_elec)
        else:
            coneMapper.SetInputConnection(cone.GetOutputPort())

        coneActor = vtk.vtkActor()
        coneActor.SetMapper(coneMapper)
        if len(colour):
            coneActor.GetProperty().SetColor(colour[0], colour[1], colour[2])
        coneActor.GetProperty().SetOpacity(opacity)

        self.actors.append(coneActor)
        self.ids.append(idx)

    def create_vectorfield_actor(self, idx=None, num_elec=None, source=None, field=None, colour=None):
        # vectors location
        points = vtk.vtkPoints()
        for p in range(len(source)):
            x = source[p]
            points.InsertPoint(p, x[0], x[1], x[2])

        sourcePolyData = vtk.vtkPolyData()
        sourcePolyData.SetPoints(points)

        # direction
        directionArray = vtk.vtkDoubleArray()
        directionArray.SetName("direction")
        directionArray.SetNumberOfComponents(3)
        directionArray.SetNumberOfTuples(sourcePolyData.GetNumberOfPoints())
        for i in range(sourcePolyData.GetNumberOfPoints()):
            vector = field[i]
            directionArray.SetTuple3(i, vector[0], vector[1], vector[2])
        sourcePolyData.GetPointData().AddArray(directionArray)
        sourcePolyData.GetPointData().SetActiveVectors("direction")

        # glyphs
        arrow = vtk.vtkArrowSource()
        glyphs = vtk.vtkGlyph3D()
        glyphs.SetSourceConnection(arrow.GetOutputPort())
        glyphs.SetInputData(sourcePolyData)
        glyphs.ScalingOn()
        glyphs.SetScaleModeToScaleByVector()
        glyphs.SetVectorModeToUseVector()
        # glyphs.OrientOn()
        # glyphs.SetScaleFactor(1)
        # glyphs.SetRange(0, 1)
        glyphs.Update()

        directionMapper = vtk.vtkPolyDataMapper()
        directionMapper.SetInputConnection(glyphs.GetOutputPort())

        directionActor = vtk.vtkActor()
        directionActor.SetMapper(directionMapper)

        if len(colour):
            directionActor.GetProperty().SetColor(colour[0], colour[1], colour[2])

        self.actors.append(directionActor)
        self.ids.append(idx)

    def create_oriented_cylinder_actor(self, x1=None, x2=None, lux=0.0, luy=0.0, colour=None, opacity=0.0):
        # Create a cylinder.
        # Cylinder height vector is (0,1,0).
        # Cylinder center is in the middle of the cylinder
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetResolution(15)
        cylinderSource.CappingOff()

        # Compute a basis
        normalizedX = [0] * 3
        normalizedY = [0] * 3
        normalizedZ = [0] * 3

        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(x2, x1, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)

        # The Z axis is an Y vector cross X
        arbitrary = [0.0, 1.0, 0.0]
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)

        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()

        # Create the direction cosine matrix
        matrix.Identity()
        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(x1)  # translate to starting point
        transform.Concatenate(matrix)  # apply direction cosines
        transform.RotateZ(-90.0)  # align cylinder to x axis
        transform.Scale(luy, length, lux)  # scale along the height vector
        transform.Translate(0, .5, 0)  # translate to start of cylinder

        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(cylinderSource.GetOutputPort())

        # Create a mapper and actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        mapper.SetInputConnection(transformPD.GetOutputPort())
        actor.SetMapper(mapper)
        if len(colour):
            actor.GetProperty().SetColor(colour[0], colour[1], colour[2])
        actor.GetProperty().SetOpacity(opacity)

        return actor