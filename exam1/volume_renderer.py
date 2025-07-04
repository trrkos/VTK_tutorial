import numpy as np
import vtk


class VolumeRenderer:
    
    def __init__(self):
        self.vtk_image = None
        self.volume_mapper = None
        self.volume_property = None
        self.volume = None
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
    def numpy_to_vtk_image(self, numpy_array: np.ndarray):
        if numpy_array.ndim != 3:
            raise ValueError("Array must be 3D")

        if numpy_array.dtype != np.uint8:
            numpy_array = numpy_array.astype(np.uint8)

        vtk_data_array = vtk.vtkUnsignedCharArray()
        vtk_data_array.SetNumberOfComponents(1)
        vtk_data_array.SetNumberOfTuples(numpy_array.size)
        
        flat_array = numpy_array.ravel(order='C')
        for i in range(len(flat_array)):
            vtk_data_array.SetValue(i, int(flat_array[i]))
        
        self.vtk_image = vtk.vtkImageData()
        self.vtk_image.SetDimensions(numpy_array.shape[::-1])
        self.vtk_image.SetSpacing(1.0, 1.0, 1.0)
        self.vtk_image.SetOrigin(0.0, 0.0, 0.0)
        self.vtk_image.GetPointData().SetScalars(vtk_data_array)

        self.vtk_image.GetPointData().GetScalars().Modified()
        self.vtk_image.Modified()
    
    def create_volume_mapper(self):
        if self.vtk_image is None:
            raise ValueError("No VTK image data available")
        
        try:
            self.volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
            self.volume_mapper.SetInputData(self.vtk_image)
        except Exception:
            self.volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
            self.volume_mapper.SetInputData(self.vtk_image)
    
    def create_color_transfer_function(self, 
                                     low_color = (0.0, 0.0, 0.0),
                                     high_color = (1.0, 1.0, 1.0),
                                     data_range = None) -> vtk.vtkColorTransferFunction:
        color_func = vtk.vtkColorTransferFunction()
        
        if data_range is None:
            if self.vtk_image:
                data_range = self.vtk_image.GetScalarRange()
            else:
                data_range = (0.0, 255.0)
        
        color_func.AddRGBPoint(data_range[0], *low_color)
        color_func.AddRGBPoint(data_range[1] * 0.25, *low_color)
        color_func.AddRGBPoint(data_range[1] * 0.5, 
                              (low_color[0] + high_color[0]) / 2,
                              (low_color[1] + high_color[1]) / 2,
                              (low_color[2] + high_color[2]) / 2)
        color_func.AddRGBPoint(data_range[1] * 0.75, *high_color)
        color_func.AddRGBPoint(data_range[1], *high_color)
        
        return color_func
    
    def create_opacity_function(self, 
                               opacity: float = 0.5,
                               isovalue: float = 128.0,
                               data_range = None) -> vtk.vtkPiecewiseFunction:
        opacity_func = vtk.vtkPiecewiseFunction()
        
        if data_range is None:
            if self.vtk_image:
                data_range = self.vtk_image.GetScalarRange()
            else:
                data_range = (0.0, 255.0)
        
        opacity_func.AddPoint(data_range[0], 0.0)
        opacity_func.AddPoint(isovalue - 10, 0.0)
        opacity_func.AddPoint(isovalue, opacity * 0.2)
        opacity_func.AddPoint(isovalue + (data_range[1] - isovalue) * 0.3, opacity * 0.5)
        opacity_func.AddPoint(isovalue + (data_range[1] - isovalue) * 0.6, opacity * 0.8)
        opacity_func.AddPoint(data_range[1], opacity)
        
        return opacity_func
    
    def create_gradient_opacity_function(self, 
                                        data_range = None) -> vtk.vtkPiecewiseFunction:
        gradient_func = vtk.vtkPiecewiseFunction()
        
        if data_range is None:
            data_range = (0.0, 50.0)
        
        gradient_func.AddPoint(0, 0.0)
        gradient_func.AddPoint(data_range[1] * 0.1, 0.0)
        gradient_func.AddPoint(data_range[1] * 0.3, 0.3)
        gradient_func.AddPoint(data_range[1] * 0.6, 0.7)
        gradient_func.AddPoint(data_range[1], 1.0)
        
        return gradient_func
    
    def create_volume_property(self, 
                              low_color = (0.0, 0.2, 0.4),
                              high_color = (1.0, 0.8, 0.6),
                              opacity: float = 0.3,
                              isovalue: float = 128.0) -> vtk.vtkVolumeProperty:
        self.volume_property = vtk.vtkVolumeProperty()
        
        if self.vtk_image:
            data_range = self.vtk_image.GetScalarRange()
            data_min, data_max = data_range
        else:
            data_min, data_max = 0.0, 255.0
        
        color_transfer = vtk.vtkColorTransferFunction()
        color_transfer.AddRGBPoint(data_min, 0.0, 0.0, 0.0)
        color_transfer.AddRGBPoint(data_min + (data_max - data_min) * 0.3, low_color[0], low_color[1], low_color[2])
        color_transfer.AddRGBPoint(data_min + (data_max - data_min) * 0.7, high_color[0], high_color[1], high_color[2])
        color_transfer.AddRGBPoint(data_max, 1.0, 1.0, 1.0)
        
        opacity_transfer = vtk.vtkPiecewiseFunction()
        opacity_transfer.AddPoint(data_min, 0.0)
        opacity_transfer.AddPoint(isovalue - abs(data_max - data_min) * 0.1, 0.0)
        opacity_transfer.AddPoint(isovalue, opacity * 0.5)
        opacity_transfer.AddPoint(isovalue + (data_max - isovalue) * 0.5, opacity * 0.8)
        opacity_transfer.AddPoint(data_max, opacity)
        
        gradient_opacity = vtk.vtkPiecewiseFunction()
        gradient_range = abs(data_max - data_min) * 0.1
        gradient_opacity.AddPoint(0, 0.0)
        gradient_opacity.AddPoint(gradient_range * 0.5, 0.3)
        gradient_opacity.AddPoint(gradient_range, 1.0)
        
        self.volume_property.SetColor(color_transfer)
        self.volume_property.SetScalarOpacity(opacity_transfer)
        self.volume_property.SetGradientOpacity(gradient_opacity)
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.ShadeOn()
        self.volume_property.SetAmbient(0.4)
        self.volume_property.SetDiffuse(0.6)
        self.volume_property.SetSpecular(0.2)
        
        return self.volume_property
    
    def create_volume(self) -> vtk.vtkVolume:
        if self.volume_mapper is None or self.volume_property is None:
            raise ValueError("Volume mapper and property must be created first")
        
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        
        return self.volume
    
    def update_volume_properties(self, 
                                low_color,
                                high_color,
                                opacity: float):
        if self.volume_property is None:
            return
            
        color_func = self.create_color_transfer_function(low_color, high_color)
        self.volume_property.SetColor(color_func)
        
        opacity_func = self.create_opacity_function(opacity)
        self.volume_property.SetScalarOpacity(opacity_func)
    
    def get_volume_info(self) -> dict:
        if self.vtk_image is None:
            return {}
        
        dimensions = self.vtk_image.GetDimensions()
        spacing = self.vtk_image.GetSpacing()
        scalar_range = self.vtk_image.GetScalarRange()
        
        return {
            'dimensions': dimensions,
            'spacing': spacing,
            'scalar_range': scalar_range,
            'memory_size': self.vtk_image.GetActualMemorySize()
        }