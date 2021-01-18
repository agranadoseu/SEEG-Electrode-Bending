"""
This class handles all XML parser related functions

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
import xml.etree.ElementTree as ET


class XMLParser:

    def __init__(self):
        return

    def indent(self, elem=None, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem=elem, level=level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def open_mps(self, file=None):
        # parse
        tree = ET.parse(file)
        root = tree.getroot()

        # get <time_series>
        ps = root.find('point_set')
        ts = ps.find('time_series')
        N = len(ts.getchildren())
        pos_array = np.zeros((N - 2, 3), dtype=np.float64)
        i = 0
        # IMPORTANT: negate x, y for EpiNav
        for contact in ts.iter('point'):
            pos_array[i, 0] = -float(contact.find('x').text)
            pos_array[i, 1] = -float(contact.find('y').text)
            pos_array[i, 2] = float(contact.find('z').text)
            i += 1

        return pos_array

    def save_points_as_mps(self, points=None, mps_file=None):
        ''' Save an array of points as an MPS file (mps) '''
        N = len(points)

        # create XML file from scratch
        root = ET.fromstring(
            "<point_set_file><file_version>0.1</file_version><point_set><time_series><time_series_id>0</time_series_id>" +
            "<Geometry3D ImageGeometry=\"false\" FrameOfReferenceID=\"0\">" +
            "<IndexToWorld type=\"Matrix3x3\" m_0_0=\"1\" m_0_1=\"0\" m_0_2=\"0\" m_1_0=\"0\" m_1_1=\"1\" m_1_2=\"0\" m_2_0=\"0\" m_2_1=\"0\" m_2_2=\"1\" />" +
            "<Offset type=\"Vector3D\" x=\"0\" y=\"0\" z=\"0\" />" +
            "<Bounds><Min type=\"Vector3D\" x=\"0.0\" y=\"0.0\" z=\"0.0\" />" +
            "<Max type=\"Vector3D\" x=\"0.0\" y=\"0.0\" z=\"0.0\" /></Bounds>" +
            "</Geometry3D>" +
            "</time_series></point_set></point_set_file>")
        tree = ET.ElementTree(root)
        root = tree.getroot()
        print('root', root)
        for child in root:
            print('tag={} attrib={}'.format(child.tag, child.attrib))

        # edit: Min, Max, insert points
        nodePointSet = root.find('point_set')
        nodeTimeSeries = nodePointSet.find('time_series')
        nodeGeometry3D = nodeTimeSeries.find('Geometry3D')
        nodeBounds = nodeGeometry3D.find('Bounds')
        print('nodeTimeSeries', nodeTimeSeries)
        nodeMin = nodeBounds.find('Min')
        print('nodeMin', nodeMin)
        nodeMax = nodeBounds.find('Max')
        print('nodeMax', nodeMax)

        # update
        for i in range(N):
            self.xml_add_point(node=nodeTimeSeries, id=i, point=points[i])
        self.xml_update_vector3d(node=nodeMin, point=np.min(points, axis=0))
        self.xml_update_vector3d(node=nodeMax, point=np.max(points, axis=0))

        # write
        self.indent(root)
        tree.write(mps_file, encoding="UTF-8", xml_declaration=True)

    def save_points_as_xml(self, name=None, ep=None, points=None, colour=None, xml_file=None):
        ''' Save an array of points as an electrode file (xmlE) '''
        number = int(''.join([i for i in name if i.isdigit()]))
        N = len(points)

        # create XML file from scratch
        root = ET.fromstring("<electrode version=\"17.04.03\">" +
                             "<planName></planName><planID>0</planID><planNumber>0</planNumber>" +
                             "<color><redValue>1</redValue><greenValue>1</greenValue><blueValue>1</blueValue><alphaValue>0.5</alphaValue></color>" +
                             "<name>test</name><electrodeNumber>0</electrodeNumber>" +
                             "<sampleNum>128</sampleNum><gmSampleNum>30</gmSampleNum><signCorrection>1</signCorrection>" +
                             "<electrodeID>1491747690</electrodeID>" +
                             "<deepStructureID>0</deepStructureID><superStructureID>0</superStructureID><placementModeID>0</placementModeID><numOfIntersections>0</numOfIntersections>" +
                             "<maxAngle>15</maxAngle><maxLength>50</maxLength><safetyMargin>3</safetyMargin><riskZone>10</riskZone><ignoreDistFromTip>0</ignoreDistFromTip>" +
                             "<lockEntry>1</lockEntry><lockTarget>1</lockTarget>" +
                             "<sorted>0</sorted><profileName></profileName><diameter>1</diameter><activeLength>0</activeLength>" +
                             "<numOfContacts>10</numOfContacts>" +
                             "<contactPointArray></contactPointArray>" +
                             "<contactColorArray/><contactSamplingRadius>2.29</contactSamplingRadius><contactSize>2.29</contactSize>" +
                             "<geometry>" +
                             "<index2worldMat><Mat00>1</Mat00><Mat01>0</Mat01><Mat02>0</Mat02><Mat10>0</Mat10><Mat11>1</Mat11><Mat12>0</Mat12><Mat20>0</Mat20><Mat21>0</Mat21><Mat22>1</Mat22></index2worldMat>" +
                             "<index2worldMatOffs><Offs0>0</Offs0><Offs1>0</Offs1><Offs2>0</Offs2></index2worldMatOffs>" +
                             "<index2worldScale><Scale0>1</Scale0><Scale1>1</Scale1><Scale2>1</Scale2></index2worldScale>" +
                             "<bounds><Bound0>0</Bound0><Bound1>1</Bound1><Bound2>0</Bound2><Bound3>1</Bound3><Bound4>0</Bound4><Bound5>1</Bound5></bounds>" +
                             "<spacing><Spacing0>1</Spacing0><Spacing1>1</Spacing1><Spacing2>1</Spacing2></spacing>" +
                             "<isImageGeom>0</isImageGeom>" +
                             "</geometry>" +
                             "<bestIndex>-1</bestIndex><targets><numOfTargets>0</numOfTargets></targets><targetIndices><numOfTargetIndices>0</numOfTargetIndices></targetIndices>" +
                             "<specifiedTargetPoint><point0>0.0</point0><point1>0.0</point1><point2>0.0</point2></specifiedTargetPoint>" +
                             "<specifiedEntryPoint><point0>0.0</point0><point1>0.0</point1><point2>0.0</point2></specifiedEntryPoint>" +
                             "</electrode>")
        tree = ET.ElementTree(root)
        root = tree.getroot()

        # edit: color, name, electrodeNumber, numOfContacts, contactPointArray, specifiedTargetPoint, specifiedEntryPoint
        nodeColour = root.find('color')
        nodeName = root.find('name')
        nodeElectrodeNumber = root.find('electrodeNumber')
        nodeNumContacts = root.find('numOfContacts')
        nodeContactPointArray = root.find('contactPointArray')
        nodeSpecifiedTargetPoint = root.find('specifiedTargetPoint')
        nodeSpecifiedEntryPoint = root.find('specifiedEntryPoint')

        # update
        self.xml_update_colour(node=nodeColour, colour=colour)
        for i in range(N):
            self.xml_add_contact_point(node=nodeContactPointArray, point=points[i])
        self.xml_update_value(node=nodeName, value=name)
        self.xml_update_value(node=nodeElectrodeNumber, value=number)
        self.xml_update_value(node=nodeNumContacts, value=N)
        if N > 0:
            self.xml_update_point(node=nodeSpecifiedTargetPoint, point=points[0])
        self.xml_update_point(node=nodeSpecifiedEntryPoint, point=ep)

        # write
        self.indent(root)
        tree.write(xml_file, encoding="UTF-8", xml_declaration=True)

    def load_electrode(self, filename=None):
        # parse interpolation
        tree = ET.parse(filename)
        root = tree.getroot()

        # get entry and target point
        ep = self.xml_get_point(parent=root, node='specifiedEntryPoint')
        tp = self.xml_get_point(parent=root, node='specifiedTargetPoint')

        # get point coordinates
        points = self.xml_get_pointarray(parent=root, node='contactPointArray')

        return ep, tp, points

    def xml_get_point(self, parent=None, node=None):
        point = parent.find(node)
        x = float(point.find('point0').text)
        y = float(point.find('point1').text)
        z = float(point.find('point2').text)
        point_array = np.array([x, y, z], dtype=np.float64)

        return point_array

    def xml_get_pointarray(self, parent=None, node=None):
        point_array = parent.find(node)
        N = len(point_array.getchildren())
        pos_array = np.zeros((N, 3), dtype=np.float64)

        i = 0
        for contact in point_array.iter('contactPoint'):
            pos_array[i, 0] = float(contact.find('point0').text)
            pos_array[i, 1] = float(contact.find('point1').text)
            pos_array[i, 2] = float(contact.find('point2').text)
            i += 1

        return pos_array

    def xml_update_point(self, node=None, point=None):
        nodePoint0 = node.find('point0')
        nodePoint1 = node.find('point1')
        nodePoint2 = node.find('point2')
        nodePoint0.text = str(point[0])
        nodePoint1.text = str(point[1])
        nodePoint2.text = str(point[2])

    def xml_update_vector3d(self, node=None, point=None):
        # <Min type=\"Vector3D\" x=\"0.0\" y=\"0.0\" z=\"0.0\" />
        node.set('x', str(point[0]))
        node.set('y', str(point[1]))
        node.set('z', str(point[2]))

    def xml_update_colour(self, node=None, colour=None):
        nodeRed = node.find('redValue')
        nodeGreen = node.find('greenValue')
        nodeBlue = node.find('blueValue')
        nodeRed.text = str(colour[0])
        nodeGreen.text = str(colour[1])
        nodeBlue.text = str(colour[2])

    def xml_add_point(self, node=None, id=None, point=None):
        # <point>
        #     <id>0</id>
        #     <specification>0</specification>
        #     <x>0.0</x>
        #     <y>0.0</y>
        #     <z>0.0</z>
        # </point>
        point_node = ET.SubElement(node, 'point')
        ET.SubElement(point_node, 'id').text = str(id)
        ET.SubElement(point_node, 'specification').text = str(0)
        ET.SubElement(point_node, 'x').text = str(point[0])
        ET.SubElement(point_node, 'y').text = str(point[1])
        ET.SubElement(point_node, 'z').text = str(point[2])

    def xml_add_contact_point(self, node=None, point=None):
        # <contactPoint>
        #  <point0>0.0</point0>
        #  <point1>0.0</point1>
        #  <point2>0.0</point2>
        # </contactPoint>
        point_node = ET.SubElement(node, 'contactPoint')
        ET.SubElement(point_node, 'point0').text = str(point[0])
        ET.SubElement(point_node, 'point1').text = str(point[1])
        ET.SubElement(point_node, 'point2').text = str(point[2])

    def xml_update_value(self, node=None, value=None):
        node.text = str(value)