import os
import sys 
sys.argv = '1'
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import tensorflow.keras.models as M 
import tensorflow.keras.backend as K
import cv2
import numpy as np 
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.util import img_as_float
import open3d as o3d
from skimage.measure import label 
import SegmentRegistration

### NEED TO INSTALL IN SLICER BEFORE IMPORTING THE MODULE ###
# pip_install('opencv-python==4.5.4.58')
# pip_install('tensorflow')
# pip_install('keras==2.6.0')
#pip_install('scikit-image')
#pip_install('open3d==0.14.1') 
# Prostate
# print packages:
# import pkg_resources
# installed_packages = pkg_resources.working_set
# installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
   # for i in installed_packages])
# print(installed_packages_list)
#To replace resliced image set IJkToRASMat (from original scan) as top transform followed by T = imageResliceForeground.GetResliceTransform()

class Prostate(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Prostate"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Prostate">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # Prostate1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='Prostate',
    sampleName='Prostate1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'Prostate1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='Prostate1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='Prostate1'
  )

  # Prostate2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='Prostate',
    sampleName='Prostate2',
    thumbnailFileName=os.path.join(iconsPath, 'Prostate2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='Prostate2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='Prostate2'
  )

#
# ProstateWidget
#

class ProstateWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    slicer.mymod = self
    self.path = os.path.dirname(os.path.abspath(__file__))
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.smooth = 1 
    self.UNetModel = M.load_model(self.path + './Resources/Models/'+'Prostate.h5', custom_objects={'dice_coef_loss': self.dice_coef_loss,'dice_coef':self.dice_coef})
    self.UNetModel.load_weights(self.path + './Resources/Models/'+'Prostate.h5', by_name=True)
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code

    self.edge = vtk.vtkFeatureEdges()
    self.edge.BoundaryEdgesOn()
    self.edge.FeatureEdgesOn() 
    self.edge.ManifoldEdgesOff()

    self.img_rows = 256
    self.img_cols = 256
    self.ijktoRasTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    self.shiftTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    self.ijktoRasTransform.SetName('IJKToRAS Transform') 
    self.ijkToRasMat = vtk.vtkMatrix4x4()
    self.shift = vtk.vtkMatrix4x4()
    self.resliceLogic = slicer.modules.volumereslicedriver.logic()
    self.Immean = None
    # self.reslicedImage = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')      
    self.prostateSeg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    self.count = 1 
    self.segLog = slicer.modules.segmentations.logic()
    self.inputVolume =None 
    self.APD = vtk.vtkAppendPolyData()
    self.firstSeg = True 
    self.inputVolume = None 
    self.setSlice = True 


  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/Prostate.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    
    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)

    # Reconstruction button
    self.ui.setSliceButton.connect('clicked(bool)', self.onsetSliceClicked)

    # Registration buttons
    self.ui.modelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onModelChanged)
    self.ui.MRVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMRVolumeChanged)
    self.ui.registerButton.connect('clicked(bool)', self.onRegClicked)
    self.ui.zoneSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onZoneChanged)
    self.ui.invertButton.connect('clicked(bool)', self.onInvertClicked)

    # Metrics dropdowns/buttons
    self.ui.GTSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onGTChanged)
    self.ui.RegZoneSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onPredChanged)
    self.ui.metricsButton.connect('clicked(bool)', self.metrics)
    # Make sure parameter node is initialized (needed for module reload)

  # Method to set the ground truth model
  def onGTChanged(self):
    self.GTNode = self.ui.GTSelector.currentNode()
  
  # Method to set the predicted model
  def onPredChanged(self):
    self.predNode = self.ui.RegZoneSelector.currentNode()

  # method to set the input volume -this will always be the Ultrasound volume
  def onInputVolumeChanged(self):
    # if self.ui.inputSelector.currentNode() is None:
    self.inputVolume = self.ui.inputSelector.currentNode() 
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    self.ui.setSliceButton.enabled=True 

  # set the MR volume 
  def onMRVolumeChanged(self):
    self.MRVolume = self.ui.MRVolumeSelector.currentNode()

    # it auto rotates 90 degrees to match with the reconstructed prostate. Might not be necessary with diff types of data
    T = vtk.vtkTransform()
    T.RotateX(-90)
    rotateTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    rotateTransform.SetAndObserveTransformToParent(T) 
    self.MRVolume.SetAndObserveTransformNodeID(rotateTransform.GetID())
    self.MRVolume.HardenTransform()
    slicer.mrmlScene.RemoveNode(rotateTransform) 

  # set the MRI prostate model
  def onModelChanged(self):
    self.modelNode = self.ui.modelSelector.currentNode() 
    self.ui.registerButton.enabled=True

    # it auto rotates 90 degrees to match with the reconstructed prostate. Might not be necessary with diff types of data
    T = vtk.vtkTransform()
    T.RotateX(-90)
    rotateTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    rotateTransform.SetAndObserveTransformToParent(T) 
    self.modelNode.SetAndObserveTransformNodeID(rotateTransform.GetID())
    self.modelNode.HardenTransform()
    slicer.mrmlScene.RemoveNode(rotateTransform) 

  # set the PZ model
  def onZoneChanged(self):
    self.pzNode = self.ui.zoneSelector.currentNode() 
    self.ui.invertButton.enabled=True

    # it auto rotates 90 degrees to match with the reconstructed prostate. Might not be necessary with diff types of data
    T = vtk.vtkTransform()
    T.RotateX(-90)
    rotateTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    rotateTransform.SetAndObserveTransformToParent(T) 
    self.pzNode.SetAndObserveTransformNodeID(rotateTransform.GetID())
    self.pzNode.HardenTransform()
    slicer.mrmlScene.RemoveNode(rotateTransform) 

  # Set up the deformable registration
  def onRegClicked(self):
    s = slicer.mrmlScene

    # make the segmentations from the models
    self.modelSeg = s.AddNewNodeByClass("vtkMRMLSegmentationNode", "MR")   #MRI segmentation
    self.usSeg = s.AddNewNodeByClass("vtkMRMLSegmentationNode", "US")      #US segmentation
    self.segLog.ImportModelToSegmentationNode(self.modelNode, self.modelSeg)
    self.segLog.ImportModelToSegmentationNode(self.finalModel, self.usSeg)

    widget = SegmentRegistration.SegmentRegistrationWidget() # this will open the segment registration widget -user input required

  
  # fix all the transforms to set the PZ on the TRUS properly
  def onInvertClicked(self):
    s = slicer.mrmlScene

    prealign = s.GetFirstNodeByName("PreAlignmentMoving2FixedLinearTransform")
    self.inputVolume.SetAndObserveTransformNodeID(None)  #remove the transform from the TRUS

    #prealign.Inverse()    
    #self.inputVolume.SetAndObserveTransformNodeID(prealign.GetID()) #shift the TRUS back to its original state
    self.finalModel.SetAndObserveTransformNodeID(prealign.GetID()) #shift the TRUS model to new coordinates

    # hide the segmentations
    displayNode = self.usSeg.GetDisplayNode()
    displayNode.SetAllSegmentsVisibility(False)
    displayNode = self.modelSeg.GetDisplayNode()
    displayNode.SetAllSegmentsVisibility(False) 

    #get and invert the transform. Apply it to the MRI model
    transform = s.GetFirstNodeByName( "Deformable Transform")
    transform.Inverse()
    #self.modelNode.SetAndObserveTransformNodeID(prealign.GetID())
    #self.modelNode.HardenTransform()
    self.modelNode.SetAndObserveTransformNodeID(transform.GetID())

    #apply to zone
    #self.pzNode.SetAndObserveTransformNodeID(prealign.GetID())
    #self.pzNode.HardenTransform()
    self.pzNode.SetAndObserveTransformNodeID(transform.GetID())


  def dice_coef_loss(self, y_true, y_pred):
    return -dice_coef(y_true, y_pred)

  def dice_coef(self, y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

  def predict(self,im):
    im = im/255. 
    self.x = np.expand_dims( cv2.resize(im,[256,256]), axis=2)
    self.out = self.UNetModel.predict(np.expand_dims(self.x, axis=0))

  # Run the reconstruction
  def onsetSliceClicked(self):
    inputVolumeNode = self.inputVolume
    observerTag = None
    spacing = inputVolumeNode.GetSpacing()
    outputSpacing = [spacing[0],spacing[1],spacing[2]]  # Millimeters/pixel
    A = slicer.util.arrayFromVolume(inputVolumeNode)

    outputExtent = [0, int(A.shape[1])-1 , 0, int(A.shape[0])-1, 0,int(spacing[2]*100*2)-1]# First and last pixel indices along each axis

    # Compute reslice transformation
    self.volumeToIjkMatrix = vtk.vtkMatrix4x4()
    inputVolumeNode.GetRASToIJKMatrix(self.volumeToIjkMatrix)
    #inputVolumeNode.GetIJKToRASDirectionMatrix(volumeToIjkMatrix)

    b = [0,0,0,0,0,0]
    inputVolumeNode.GetBounds(b) 

    T = vtk.vtkTransform()
    #T.Translate(-b[0]/2,-b[2]/2, -b[-1]/2) 
    T.RotateX(-90)
    centerTransform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    centerTransform.SetAndObserveTransformToParent(T) 

    inputVolumeNode.SetAndObserveTransformNodeID(centerTransform.GetID())


    self.sliceToRasTransform = vtk.vtkTransform()
    self.sliceToRasTransform.Translate(b[0], -b[-1], 0)


    self.sliceToIjkTransform = vtk.vtkTransform()
    self.sliceToIjkTransform.Concatenate(self.volumeToIjkMatrix)
    self.sliceToIjkTransform.Concatenate(self.sliceToRasTransform)

    # Use MRML node to modify transform in GUI

    sliceToRasNode = slicer.mrmlScene.GetFirstNodeByName("SliceToRas")
    if sliceToRasNode is None:
      sliceToRasNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "SliceToRas")

    sliceToRasNode.SetAndObserveTransformToParent(self.sliceToRasTransform)

    # Run reslice to produce output image

    self.reslice = vtk.vtkImageReslice()
    self.reslice.SetInputData(inputVolumeNode.GetImageData())
    self.reslice.SetResliceTransform(self.sliceToIjkTransform)
    self.reslice.SetInterpolationModeToLinear()
    self.reslice.SetOutputOrigin(0.0, 0.0, 0.0)  # Must keep zero so transform overlays slice with volume
    self.reslice.SetOutputSpacing(outputSpacing)
    self.reslice.SetOutputDimensionality(2)
    self.reslice.SetOutputExtent(outputExtent)
    self.reslice.SetBackgroundLevel(0)
    self.reslice.Update()
    self.reslice.GetOutput().SetSpacing(1,1,1)  # Spacing will be set on MRML node to let Slicer know

    # To allow re-run of this script, try to reuse exisiting node before creating a new one

    outputNode = slicer.mrmlScene.GetFirstNodeByName("OutputVolume")
    if outputNode is None:
      outputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "OutputVolume")

    outputImageData = self.reslice.GetOutput()
    outputNode.SetAndObserveImageData(self.reslice.GetOutput())
    outputNode.SetSpacing(outputSpacing)
    outputNode.CreateDefaultDisplayNodes()

    # Transform output image so it is aligned with original volume

    outputNode.SetAndObserveTransformNodeID(sliceToRasNode.GetID())

    # Show and follow output image in red slice view

    redSliceWidget = slicer.app.layoutManager().sliceWidget("Red")
    redSliceWidget.sliceController().setSliceVisible(True)
    redSliceWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(outputNode.GetID())
    driver = slicer.modules.volumereslicedriver.logic()
    redView = slicer.util.getNode('vtkMRMLSliceNodeRed')
    driver.SetModeForSlice(driver.MODE_TRANSVERSE, redView)
    driver.SetDriverForSlice(outputNode.GetID(), redView)
    if observerTag is not None:
      sliceToRasNode.RemoveObserver(observerTag)
      observerTag = None

    observerTag = sliceToRasNode.AddObserver(sliceToRasNode.TransformModifiedEvent, self.updateReslice)
    self.applyRotation(outputNode, 0,sliceToRasNode)
    self.sliceToRasNode = sliceToRasNode
    self.performRecon() 
    inputVolumeNode.HardenTransform()

  def updateReslice(self, event, caller):
    try:
      slicer.app.pauseRender()
      
      inputTransformId = self.inputVolume.GetTransformNodeID()
      if inputTransformId is not None:
        inputTransformNode = slicer.mrmlScene.GetNodeByID(inputTransformId)
        rasToVolumeMatrix = vtk.vtkMatrix4x4()
        inputTransformNode.GetMatrixTransformFromWorld(rasToVolumeMatrix)
        self.sliceToIjkTransform.Identity()
        self.sliceToIjkTransform.Concatenate(self.volumeToIjkMatrix)
        self.sliceToIjkTransform.Concatenate(rasToVolumeMatrix)
        self.sliceToIjkTransform.Concatenate(self.sliceToRasTransform)
      
      self.reslice.Update()
      self.reslice.GetOutput().SetSpacing(1,1,1)
    finally:
      slicer.app.resumeRender()

  def applyRotation(self,outputNode,deg,sliceToRasNode): 
    b = [0,0,0,0,0,0]
    self.inputVolume.GetBounds(b) 
    rotate = vtk.vtkTransform()
    rotate.RotateY(deg)
    rotate.Translate(b[0], -b[-1], 0)
    mat = vtk.vtkMatrix4x4()
    rotate.GetMatrix(mat)
    sliceToRasNode.SetMatrixTransformToParent(mat) 
    outputNode.GetDisplayNode().SetWindowLevelMinMax(0,178)

  def getSegmentationFromSlice(self, outputNode, sliceToRasNode,deg):
    self.npImage = np.squeeze(slicer.util.arrayFromVolume(outputNode))
    b = [0,0,0,0,0,0]
    outputNode.GetBounds(b) 
    # self.npImage = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
    self.segmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    self.segmentation.SetName('seg'+str(deg))
    self.padL = int(np.ceil((510 - self.npImage.shape[0]) / 2)) - np.mod(self.npImage.shape[0], 2)
    self.padR = int(np.ceil((510 - self.npImage.shape[0]) / 2))
    self.padU = int(np.ceil((788 - self.npImage.shape[1]) / 2)) - np.mod(self.npImage.shape[1], 2)
    self.padD = int(np.ceil((788 - self.npImage.shape[1]) / 2))
    self.newIm = np.pad(self.npImage, ((self.padL, self.padR), (self.padU, self.padD)), mode='constant')
    self.newIm = img_as_float(self.newIm)
    self.Immean = np.mean(self.newIm)
    self.Imstd = np.std(self.newIm)
    self.newIm -= self.Immean
    self.newIm /= self.Imstd
    rows = cols = 256
    rimg = resize(self.newIm, (rows, cols), preserve_range=True)
    rimgs = np.expand_dims(rimg, axis=0)
    rimgs = np.expand_dims(rimgs, axis=3)
    out = self.UNetModel.predict(rimgs)
    self.o = np.squeeze(out)
    self.mskout = resize(self.o, (510, 788), preserve_range=True)
    slicer.util.updateVolumeFromArray(self.segmentation, np.expand_dims(self.mskout, axis=0)) #Generating the label map
    self.shift.SetElement(0, 3, - self.padU)
    self.shift.SetElement(1, 3, - self.padL)
    self.shiftTransform.SetMatrixTransformToParent(self.shift)
    self.shiftTransform.SetName('Shift')
    mat = vtk.vtkMatrix4x4()
    outputNode.GetIJKToRASMatrix(mat)
    self.ijktoRasTransform.SetMatrixTransformToParent(mat)
    self.ijktoRasTransform.SetAndObserveTransformNodeID(sliceToRasNode.GetID())
    self.shiftTransform.SetAndObserveTransformNodeID(self.ijktoRasTransform.GetID()) 
    self.segmentation.SetAndObserveTransformNodeID(self.shiftTransform.GetID())
    self.segmentation.HardenTransform()
    self.segLog.ImportLabelmapToSegmentationNode(self.segmentation, self.prostateSeg)
    self.segLog.ExportAllSegmentsToModels(self.prostateSeg, 0)
    Mods = slicer.mrmlScene.GetNodesByClassByName('vtkMRMLModelNode', 'seg'+str(deg))
    self.segMod = Mods.GetItemAsObject(0)
    polydata = self.segMod.GetPolyData()
    self.APD.AddInputData(polydata)
    self.APD.Update()
    slicer.mrmlScene.RemoveNode(self.segMod)
    slicer.mrmlScene.RemoveNode(self.segmentation)
    self.prostateSeg.RemoveSegment('seg'+str(deg))

  def performRecon(self):
    outputNode = slicer.mrmlScene.GetFirstNodeByName("OutputVolume")
    sliceToRasNode = slicer.mrmlScene.GetFirstNodeByName("SliceToRas")
    
    for i in range(0,195,15):
      self.applyRotation(outputNode,i,sliceToRasNode)
      self.getSegmentationFromSlice(outputNode, sliceToRasNode,i)
    self.onModelButtonClicked() 
    slicer.mrmlScene.RemoveNode(self.ijktoRasTransform)
    slicer.mrmlScene.RemoveNode(self.shiftTransform)
    slicer.mrmlScene.RemoveNode(outputNode)
    slicer.mrmlScene.RemoveNode(sliceToRasNode) 
    slicer.mrmlScene.RemoveNode(self.prostateSeg) 

    self.ui.registerButton.enabled=True 

  
  def onModelButtonClicked(self):
    out = self.APD.GetOutput()
    self.edge.SetInputData(out)
    self.edge.Update()
    depth = 3 
    bound = self.edge.GetOutput()
    normals = bound.GetPointData().GetNormals()
    pts = bound.GetPoints().GetData()
    pts_np = vtk.util.numpy_support.vtk_to_numpy(pts)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, width=0, scale=1.1, linear_fit=False)[0]
    # o3d.visualization.draw_geometries([poisson_mesh])

    outputPath = self.path + './/Resources//Models//'
    poisson_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(poisson_mesh)
    o3d.io.write_triangle_mesh(outputPath + "mesh.stl", poisson_mesh)
    self.finalModel = slicer.util.loadModel(outputPath+"mesh.stl")
    os.remove(outputPath+"mesh.stl")
    self.finalModel.SetName('Reconstructed Prostate') 
    Tnode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode') 
    rot = vtk.vtkTransform()
    rot.RotateX(-180)
    rot.RotateY(-180) 
    Tnode.SetAndObserveTransformToParent(rot)
    self.finalModel.SetAndObserveTransformNodeID(Tnode.GetID())
    self.finalModel.HardenTransform()
    slicer.mrmlScene.RemoveNode(Tnode) 
    DN = self.finalModel.GetDisplayNode() 
    DN.SetSliceIntersectionVisibility(1)
    # DN.SetSliceIntersectionOpacity(0.25)
    slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceNode().SetSliceVisible(True)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)


  # get quantitative metrics between a GT model and a predicted model
  def metrics(self):
    
    Predictedmodel = self.predNode
    GroundTruthmodel = self.GTNode

    #Generate Label map 
    Im = self.inputVolume
    A = slicer.util.arrayFromVolume(Im)

    # get the labelmap then array from the predicted model
    seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    segLog = slicer.modules.segmentations.logic()
    seg.SetReferenceImageGeometryParameterFromVolumeNode(Im)
    segLog.ImportModelToSegmentationNode(Predictedmodel,seg)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(Im)
    LM = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    segLog.ExportVisibleSegmentsToLabelmapNode(seg,LM,Im)
    self.Predict = slicer.util.arrayFromVolume(LM)
    Predict = self.Predict

    # get the labelmap then array from the ground truth model
    seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    segLog = slicer.modules.segmentations.logic()
    seg.SetReferenceImageGeometryParameterFromVolumeNode(Im)
    segLog.ImportModelToSegmentationNode(GroundTruthmodel,seg)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(Im)
    LM = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    segLog.ExportVisibleSegmentsToLabelmapNode(seg,LM,Im)
    GT = slicer.util.arrayFromVolume(LM)

    # compute the metrics based on TP, TN, FP, FN
    self.true_pos = np.logical_and(GT==1,Predict==1)
    self.true_neg = np.logical_and(GT==0,Predict==0)
    self.false_pos = np.logical_and(GT==1,Predict==0)
    self.false_neg = np.logical_and(GT==0,Predict==1)
    self.true_pos = np.sum(self.true_pos)
    self.true_neg = np.sum(self.true_neg)
    self.false_pos = np.sum(self.false_pos)
    self.false_neg= np.sum(self.false_neg)
    self.recall = self.true_pos / (self.true_pos + self.false_neg)
    self.specificity = self.true_neg / (self.true_neg + self.false_pos)
    self.precision = self.true_pos / (self.true_pos + self.false_pos)

    intersection = np.logical_and(GT,Predict)

    # compute dice score
    self.dice_score = 2.*intersection.sum()/(GT.sum()+Predict.sum())

    # display the stats in the python interpreter
    print("Dice:", self.dice_score)
    print("Recall:", self.recall)
    print("Specificity:", self.specificity)
    print("Precision:", self.precision)
