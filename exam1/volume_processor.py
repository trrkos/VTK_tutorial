import SimpleITK as sitk
import numpy as np
import os


class VolumeProcessor:
    def __init__(self):
        self.volume = None
        self.processed_volume = None
        
    def load_dicom_series(self, dicom_directory: str) -> sitk.Image:
        if not os.path.exists(dicom_directory):
            raise FileNotFoundError(f"Directory {dicom_directory} not found")
            
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
        
        if dicom_names:
            reader.SetFileNames(dicom_names)
            try:
                self.volume = reader.Execute()
                return True
            except Exception:
                pass
        
        dcm_files = [f for f in os.listdir(dicom_directory) if f.lower().endswith('.dcm')]
        if not dcm_files:
            raise ValueError(f"No DICOM files found in {dicom_directory}")
        
        dcm_file = os.path.join(dicom_directory, dcm_files[0])
        try:
            single_reader = sitk.ImageFileReader()
            single_reader.SetFileName(dcm_file)
            self.volume = single_reader.Execute()
            return True
        except Exception as e:
            raise RuntimeError(f"Error reading DICOM: {e}")
    
    def apply_gaussian_smoothing(self, sigma: float = 1.0) -> sitk.Image:
        if self.volume is None:
            raise ValueError("No volume loaded")
        
        volume_to_process = self.volume
        
        original_spacing = volume_to_process.GetSpacing()
        original_origin = volume_to_process.GetOrigin()
        
        if volume_to_process.GetDimension() > 3:
            original_spacing = volume_to_process.GetSpacing()
            original_origin = volume_to_process.GetOrigin()
            
            if len(original_spacing) > 3:
                original_spacing = original_spacing[:3]
            if len(original_origin) > 3:
                original_origin = original_origin[:3]
            
            array = sitk.GetArrayFromImage(volume_to_process)
            
            while len(array.shape) > 3:
                if array.shape[0] == 1:
                    array = array[0]
                elif array.shape[-1] == 1:
                    array = array[..., 0]
                else:
                    array = array[0]
            
            while len(array.shape) > 3:
                if array.shape[-1] <= 4:
                    array = array[..., 0]
                else:
                    array = array[0]
            
            volume_to_process = sitk.GetImageFromArray(array)
            
            if len(original_spacing) == 3:
                volume_to_process.SetSpacing(original_spacing)
            if len(original_origin) == 3:
                volume_to_process.SetOrigin(original_origin)
        
        if volume_to_process.GetPixelID() != sitk.sitkFloat32:
            try:
                volume_to_process = sitk.Cast(volume_to_process, sitk.sitkFloat32)
            except Exception:
                array = sitk.GetArrayFromImage(volume_to_process)
                array = array.astype(np.float32)
                volume_to_process = sitk.GetImageFromArray(array)
                try:
                    if len(original_spacing) == 3:
                        volume_to_process.SetSpacing(original_spacing)
                    if len(original_origin) == 3:
                        volume_to_process.SetOrigin(original_origin)
                except:
                    pass
        
        size = volume_to_process.GetSize()
        min_dimension = min(size)
        
        if min_dimension < 4:
            return volume_to_process
        
        try:
            smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
            smoothing_filter.SetSigma(sigma)
            smoothed = smoothing_filter.Execute(volume_to_process)
            return smoothed
        except Exception:
            return volume_to_process
    
    def apply_clahe(self, input_volume: sitk.Image = None, clip_limit: float = 2.0, 
                   tile_grid_size: tuple = (8, 8, 8)) -> sitk.Image:
        volume_to_process = input_volume if input_volume is not None else self.volume
        
        if volume_to_process is None:
            raise ValueError("No volume loaded")
        
        original_spacing = volume_to_process.GetSpacing()
        original_origin = volume_to_process.GetOrigin()
        
        if volume_to_process.GetDimension() > 3:
            original_spacing = volume_to_process.GetSpacing()
            original_origin = volume_to_process.GetOrigin()
            
            if len(original_spacing) > 3:
                original_spacing = original_spacing[:3]
            if len(original_origin) > 3:
                original_origin = original_origin[:3]
            
            array = sitk.GetArrayFromImage(volume_to_process)
            
            while len(array.shape) > 3:
                if array.shape[0] == 1:
                    array = array[0]
                elif array.shape[-1] == 1:
                    array = array[..., 0]
                else:
                    array = array[0]
            
            while len(array.shape) > 3:
                if array.shape[-1] <= 4:
                    array = array[..., 0]
                else:
                    array = array[0]
            
            volume_to_process = sitk.GetImageFromArray(array)
            
            if len(original_spacing) == 3:
                volume_to_process.SetSpacing(original_spacing)
            if len(original_origin) == 3:
                volume_to_process.SetOrigin(original_origin)
            
        rescale_filter = sitk.RescaleIntensityImageFilter()
        rescale_filter.SetOutputMinimum(0)
        rescale_filter.SetOutputMaximum(255)
        rescaled = rescale_filter.Execute(volume_to_process)
        
        pixel_type = rescaled.GetPixelIDTypeAsString()
        
        if 'vector' in pixel_type.lower():
            array = sitk.GetArrayFromImage(rescaled)
            
            if len(array.shape) > 3:
                array = array[..., 0]
            
            rescaled = sitk.GetImageFromArray(array.astype(np.uint8))
            rescaled.SetSpacing(volume_to_process.GetSpacing())
            rescaled.SetOrigin(volume_to_process.GetOrigin())
        else:
            try:
                rescaled = sitk.Cast(rescaled, sitk.sitkUInt8)
            except Exception:
                array = sitk.GetArrayFromImage(rescaled)
                rescaled = sitk.GetImageFromArray(array.astype(np.uint8))
                rescaled.SetSpacing(volume_to_process.GetSpacing())
                rescaled.SetOrigin(volume_to_process.GetOrigin())
        
        clahe_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
        normalized_clip_limit = min(1.0, max(0.1, clip_limit / 10.0))
        clahe_filter.SetAlpha(normalized_clip_limit)
        clahe_filter.SetBeta(normalized_clip_limit)
        clahe_result = clahe_filter.Execute(rescaled)
        
        return clahe_result
    
    def process_volume(self, gaussian_sigma: float = 1.0, 
                      clahe_clip_limit: float = 2.0,
                      use_clahe: bool = True) -> np.ndarray:
        if self.volume is None:
            raise ValueError("No volume loaded")
            
        processed = self.apply_gaussian_smoothing(gaussian_sigma)
        
        if use_clahe:
            processed = self.apply_clahe(processed, clahe_clip_limit)
        
        self.processed_volume = sitk.GetArrayFromImage(processed)
        
        return self.processed_volume
    
    def get_volume_info(self) -> dict:
        if self.volume is None:
            return {}
        
        volume_info = {
            'size': self.volume.GetSize(),
            'spacing': self.volume.GetSpacing(),
            'origin': self.volume.GetOrigin(),
            'direction': self.volume.GetDirection(),
            'pixel_type': self.volume.GetPixelIDTypeAsString(),
            'dimensions': self.volume.GetDimension()
        }
        
        try:
            volume_for_stats = self.volume
            
            original_spacing = volume_for_stats.GetSpacing()
            original_origin = volume_for_stats.GetOrigin()
            
            if volume_for_stats.GetDimension() > 3:
                original_spacing = volume_for_stats.GetSpacing()
                original_origin = volume_for_stats.GetOrigin()
                
                if len(original_spacing) > 3:
                    original_spacing = original_spacing[:3]
                if len(original_origin) > 3:
                    original_origin = original_origin[:3]
                
                array = sitk.GetArrayFromImage(volume_for_stats)
                
                while len(array.shape) > 3:
                    if array.shape[0] == 1:
                        array = array[0]
                    elif array.shape[-1] == 1:
                        array = array[..., 0]
                    else:
                        array = array[0]
                
                while len(array.shape) > 3:
                    if array.shape[-1] <= 4:
                        array = array[..., 0]
                    else:
                        array = array[0]
                
                volume_for_stats = sitk.GetImageFromArray(array)
                
                if len(original_spacing) == 3:
                    volume_for_stats.SetSpacing(original_spacing)
                if len(original_origin) == 3:
                    volume_for_stats.SetOrigin(original_origin)
            
            stats_filter = sitk.StatisticsImageFilter()
            stats_filter.Execute(volume_for_stats)
            
            volume_info.update({
                'min_value': stats_filter.GetMinimum(),
                'max_value': stats_filter.GetMaximum(),
                'mean_value': stats_filter.GetMean(),
                'std_value': stats_filter.GetSigma()
            })
        except Exception:
            volume_info.update({
                'min_value': 'N/A',
                'max_value': 'N/A',
                'mean_value': 'N/A',
                'std_value': 'N/A'
            })
            
        return volume_info 