import vtk
import numpy as np
import os


class MeshExtractor:
    def __init__(self):
        self.vtk_image_data = None
        self.mesh_data = None
        self.original_mesh_data = None
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
    def numpy_to_vtk_image(self, numpy_array: np.ndarray, 
                          spacing: tuple = (1.0, 1.0, 1.0),
                          origin: tuple = (0.0, 0.0, 0.0)):
        if numpy_array.ndim != 3:
            raise ValueError("Array must be 3D")
        
        if numpy_array.dtype != np.uint8:
            numpy_array = numpy_array.astype(np.uint8)
        
        numpy_array = np.transpose(numpy_array, (2, 1, 0))
        
        flat_array = numpy_array.ravel(order='F')
        vtk_array = vtk.vtkUnsignedCharArray()
        vtk_array.SetNumberOfTuples(len(flat_array))
        for i, val in enumerate(flat_array):
            vtk_array.SetValue(i, val)
        
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(numpy_array.shape)
        image_data.SetSpacing(spacing)
        image_data.SetOrigin(origin)
        image_data.GetPointData().SetScalars(vtk_array)
        
        self.vtk_image_data = image_data
        return image_data
    
    def extract_isosurface(self, isovalue: float = 128.0):
        if self.vtk_image_data is None:
            raise ValueError("No VTK image data available")
        
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(self.vtk_image_data)
        marching_cubes.SetValue(0, isovalue)
        marching_cubes.Update()
        
        self.mesh_data = marching_cubes.GetOutput()
        return self.mesh_data
    
    def smooth_mesh(self, iterations: int = 15, relaxation_factor: float = 0.1):
        if self.mesh_data is None:
            raise ValueError("No mesh data available")
        
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(self.mesh_data)
        smoother.SetNumberOfIterations(iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        
        self.mesh_data = smoother.GetOutput()
        return self.mesh_data
    
    def create_mesh_actor(self, color: tuple = (1.0, 0.8, 0.6), 
                         opacity: float = 1.0):
        if self.mesh_data is None:
            raise ValueError("No mesh data available")
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        
        return actor
    
    def load_obj_file(self, obj_file_path: str):
        if not os.path.exists(obj_file_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_file_path}")
        
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_file_path)
        reader.Update()
        
        self.mesh_data = reader.GetOutput()
        
        if self.mesh_data.GetNumberOfPoints() == 0:
            raise ValueError(f"OBJ file contains no geometry: {obj_file_path}")
        
        return self.mesh_data
    
    def get_mesh_info(self) -> dict:
        if self.mesh_data is None:
            return {}
        
        return {
            'points': self.mesh_data.GetNumberOfPoints(),
            'cells': self.mesh_data.GetNumberOfCells(),
            'memory_mb': self.mesh_data.GetActualMemorySize() / 1024.0
        } 