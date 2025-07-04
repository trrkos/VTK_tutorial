import PySimpleGUI as sg
import vtk
import os

from volume_processor import VolumeProcessor
from mesh_extractor import MeshExtractor
from volume_renderer import VolumeRenderer


class VTKWidget:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        self.render_window.SetWindowName("3D Tomography Viewer")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        self.current_actor = None
        self.current_volume = None
        self.window_shown = False
        
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        
    def add_actor(self, actor):
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
        if self.current_volume:
            self.renderer.RemoveVolume(self.current_volume)
            
        self.current_actor = actor
        self.current_volume = None
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        
    def add_volume(self, volume):
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
        if self.current_volume:
            self.renderer.RemoveVolume(self.current_volume)
            
        self.current_volume = volume
        self.current_actor = None
        self.renderer.AddVolume(volume)
        self.renderer.ResetCamera()
        
    def render(self):
        self.renderer.ResetCamera()
        self.render_window.Render()
        
        if not self.render_window.GetNeverRendered():
            self.render_window.Render()
        
    def show_window(self):
        if not self.window_shown:
            self.render_window.Render()
            self.interactor.Initialize()
            self.window_shown = True
        
        self.render_window.Render()
        
    def start_interaction(self):
        if not self.window_shown:
            self.show_window()
        self.interactor.Start()
        
    def get_render_window(self):
        return self.render_window


class TomographyGUI:
    def __init__(self):
        self.volume_processor = VolumeProcessor()
        self.mesh_extractor = MeshExtractor()
        self.volume_renderer = VolumeRenderer()
        self.vtk_widget = VTKWidget()
        
        self.current_data = None
        self.current_obj_file = None
        self.render_mode = "Volume"
        
        self.gaussian_sigma = 1.0
        self.clahe_clip_limit = 2.0
        self.isovalue = 128.0
        self.low_color = (0.0, 0.2, 0.4)
        self.high_color = (1.0, 0.8, 0.6)
        self.opacity = 0.3
        
        sg.theme('DarkGrey11')
        
    def create_layout(self):
        header = [
            [sg.Text('3D Tomography Processing', 
                    font=('Arial', 16, 'bold'), justification='center', expand_x=True)],
            [sg.HSeparator()]
        ]
        
        model_presets = [
            [sg.Frame('Model Presets', [
                [sg.Button('Bunny', key='-PRESET_BUNNY-', size=(12, 1)),
                 sg.Button('Cow', key='-PRESET_COW-', size=(12, 1)),
                 sg.Button('Dragon', key='-PRESET_DRAGON-', size=(12, 1)),
                 sg.Button('Suzanne', key='-PRESET_SUZANNE-', size=(12, 1)),
                 sg.Button('Teapot', key='-PRESET_TEAPOT-', size=(12, 1))]
            ], expand_x=True)],
            [sg.Frame('DICOM Data', [
                [sg.Button('Brain MRI', key='-DICOM_BRAIN-', size=(12, 1)),
                 sg.Button('Multi Layer', key='-DICOM_MULTI-', size=(12, 1)),
                 sg.Text('', size=(12, 1))]
            ], expand_x=True)],
            [sg.HSeparator()]
        ]
        
        left_section = [
            [sg.Frame('Data Loading', [
                [sg.Text('DICOM Folder:', size=(12, 1)), 
                 sg.Input(default_text='real_dicom', key='-DICOM_DIR-', size=(25, 1)), 
                 sg.FolderBrowse('Browse', size=(8, 1))],
                [sg.Text('OBJ File:', size=(12, 1)), 
                 sg.Input(default_text='data/bunny.obj', key='-OBJ_FILE-', size=(25, 1)), 
                 sg.FileBrowse('Browse', file_types=(("OBJ Files", "*.obj"),), size=(8, 1))],
                [sg.Button('Load DICOM', key='-LOAD_DICOM-', size=(18, 1)),
                 sg.Button('Load OBJ', key='-LOAD_OBJ-', size=(18, 1))]
            ], expand_x=True)],
            
            [sg.Frame('Processing Parameters', [
                [sg.Text('Gaussian σ:'), sg.Text('1.0', key='-SIGMA_VAL-', size=(8, 1), justification='right')],
                [sg.Slider(range=(0.1, 5.0), default_value=1.0, resolution=0.1, 
                          orientation='h', key='-GAUSSIAN_SIGMA-', size=(35, 15), enable_events=True)],
                
                [sg.Text('CLAHE Clip:'), sg.Text('2.0', key='-CLAHE_VAL-', size=(8, 1), justification='right')],
                [sg.Slider(range=(0.5, 10.0), default_value=2.0, resolution=0.1, 
                          orientation='h', key='-CLAHE_CLIP-', size=(35, 15), enable_events=True)],
                
                [sg.Text('Isovalue:'), sg.Text('128', key='-ISO_VAL-', size=(8, 1), justification='right')],
                [sg.Slider(range=(1, 255), default_value=128, resolution=1, 
                          orientation='h', key='-ISOVALUE-', size=(35, 15), enable_events=True)]
            ], expand_x=True)]
        ]
        
        right_section = [
            [sg.Frame('Rendering Parameters', [
                [sg.Text('Render Mode:')],
                [sg.Radio('Volume', 'RENDER_MODE', default=True, key='-VOLUME_MODE-'),
                 sg.Radio('Mesh', 'RENDER_MODE', key='-MESH_MODE-')],
                
                [sg.HSeparator()],
                
                [sg.Text('Low Color (R,G,B):')],
                [sg.Text('R:'), sg.Slider(range=(0, 1), default_value=0.0, resolution=0.01, 
                         orientation='h', key='-LOW_R-', size=(10, 15)),
                 sg.Text('G:'), sg.Slider(range=(0, 1), default_value=0.2, resolution=0.01, 
                         orientation='h', key='-LOW_G-', size=(10, 15)),
                 sg.Text('B:'), sg.Slider(range=(0, 1), default_value=0.4, resolution=0.01, 
                         orientation='h', key='-LOW_B-', size=(10, 15))],
                
                [sg.Text('High Color (R,G,B):')],
                [sg.Text('R:'), sg.Slider(range=(0, 1), default_value=1.0, resolution=0.01, 
                         orientation='h', key='-HIGH_R-', size=(10, 15)),
                 sg.Text('G:'), sg.Slider(range=(0, 1), default_value=0.8, resolution=0.01, 
                         orientation='h', key='-HIGH_G-', size=(10, 15)),
                 sg.Text('B:'), sg.Slider(range=(0, 1), default_value=0.6, resolution=0.01, 
                         orientation='h', key='-HIGH_B-', size=(10, 15))],
                
                [sg.Text('Opacity:'), sg.Text('0.30', key='-OPACITY_VAL-', size=(8, 1), justification='right')],
                [sg.Slider(range=(0.0, 1.0), default_value=0.3, resolution=0.01, 
                          orientation='h', key='-OPACITY-', size=(35, 15), enable_events=True)]
            ], expand_x=True)]
        ]
        
        bottom_section = [
            [sg.HSeparator()],
            [sg.Button('Apply', key='-APPLY-', size=(15, 2), button_color=('white', 'green')),
             sg.Button('Reset', key='-RESET-', size=(15, 2), button_color=('white', 'orange')),
             sg.Text('', expand_x=True)],
            
            [sg.HSeparator()],
            
            [sg.Frame('Model Info', [
                [sg.Multiline(key='-INFO-', size=(80, 6), disabled=True, 
                            background_color='#2F4F4F', text_color='white')]
            ], expand_x=True)]
        ]
        
        layout = header + model_presets + [
            [sg.Column(left_section, vertical_alignment='top', expand_x=True),
             sg.VSeparator(),
             sg.Column(right_section, vertical_alignment='top', expand_x=True)]
        ] + bottom_section
        
        return layout
    
    def load_dicom_data(self, dicom_dir: str):
        try:
            self.volume_processor.load_dicom_series(dicom_dir)
            info = self.volume_processor.get_volume_info()
            self.update_info_display(f"DICOM loaded:\n{self.format_volume_info(info)}")
            return True
        except Exception:
            sg.popup_error("DICOM loading error")
            return False
    
    def load_obj_data(self, obj_file: str):
        try:
            self.mesh_extractor.load_obj_file(obj_file)
            
            if self.mesh_extractor.mesh_data:
                original_mesh = vtk.vtkPolyData()
                original_mesh.DeepCopy(self.mesh_extractor.mesh_data)
                self.mesh_extractor.original_mesh_data = original_mesh
                self.current_obj_file = obj_file
            
            info = self.mesh_extractor.get_mesh_info()
            model_name = os.path.basename(obj_file).replace('.obj', '').upper()
            self.update_info_display(f"Model '{model_name}' loaded!\n\n{self.format_mesh_info(info)}")
            return True
        except Exception:
            sg.popup_error("OBJ loading error")
            return False
    
    def process_and_render(self):
        if self.volume_processor.volume is None and self.mesh_extractor.mesh_data is None:
            sg.popup_error("Load data first!")
            return
            
        try:
            if self.render_mode == "Volume" and self.volume_processor.volume is not None:
                self.render_volume()
            elif self.render_mode == "Mesh":
                if self.mesh_extractor.mesh_data is not None:
                    self.render_existing_mesh()
                elif self.volume_processor.volume is not None:
                    self.render_mesh_from_volume()
        except Exception:
            sg.popup_error("Rendering error")
    
    def render_volume(self):
        processed_data = self.volume_processor.process_volume(
            gaussian_sigma=self.gaussian_sigma,
            clahe_clip_limit=self.clahe_clip_limit
        )
        
        self.volume_renderer.numpy_to_vtk_image(processed_data)
        
        self.volume_renderer.create_volume_mapper()
        self.volume_renderer.create_volume_property(
            low_color=self.low_color,
            high_color=self.high_color,
            opacity=self.opacity,
            isovalue=self.isovalue
        )
        volume = self.volume_renderer.create_volume()
        
        self.vtk_widget.add_volume(volume)
        self.vtk_widget.render()
        self.vtk_widget.show_window()
        
        self.update_info_display(f"Volume rendering complete\n\n"
                               f"Parameters:\n"
                               f"• Sigma: {self.gaussian_sigma}\n"
                               f"• CLAHE: {self.clahe_clip_limit}\n"
                               f"• Isovalue: {self.isovalue}\n"
                               f"• Opacity: {self.opacity}")
    
    def render_mesh_from_volume(self):
        processed_data = self.volume_processor.process_volume(
            gaussian_sigma=self.gaussian_sigma,
            clahe_clip_limit=self.clahe_clip_limit
        )
        
        self.mesh_extractor.numpy_to_vtk_image(processed_data)
        self.mesh_extractor.extract_isosurface(self.isovalue)
        self.mesh_extractor.smooth_mesh()
        
        actor = self.mesh_extractor.create_mesh_actor(
            color=self.high_color,
            opacity=self.opacity
        )
        
        self.vtk_widget.add_actor(actor)
        self.vtk_widget.render()
        self.vtk_widget.show_window()
        
        mesh_info = self.mesh_extractor.get_mesh_info()
        self.update_info_display(f"Mesh rendering complete\n"
                               f"Isovalue: {self.isovalue}\n"
                               f"{self.format_mesh_info(mesh_info)}")
    
    def render_existing_mesh(self):
        if hasattr(self.mesh_extractor, 'original_mesh_data') and self.mesh_extractor.original_mesh_data:
            self.mesh_extractor.mesh_data = vtk.vtkPolyData()
            self.mesh_extractor.mesh_data.DeepCopy(self.mesh_extractor.original_mesh_data)
        
        if self.gaussian_sigma > 0.1:
            num_iterations = int(self.gaussian_sigma * 10)
            smoothing_filter = vtk.vtkSmoothPolyDataFilter()
            smoothing_filter.SetInputData(self.mesh_extractor.mesh_data)
            smoothing_filter.SetNumberOfIterations(num_iterations)
            smoothing_filter.SetRelaxationFactor(0.1)
            smoothing_filter.Update()
            self.mesh_extractor.mesh_data = smoothing_filter.GetOutput()
        
        actor = self.mesh_extractor.create_mesh_actor(
            color=self.high_color,
            opacity=self.opacity
        )
        
        self.vtk_widget.add_actor(actor)
        self.vtk_widget.render()
        self.vtk_widget.show_window()
        
        mesh_info = self.mesh_extractor.get_mesh_info()
        self.update_info_display(f"Existing mesh rendered\n"
                               f"Smoothing: {self.gaussian_sigma}\n"
                               f"{self.format_mesh_info(mesh_info)}")
    
    def update_parameters_from_gui(self, values):
        self.gaussian_sigma = values['-GAUSSIAN_SIGMA-']
        self.clahe_clip_limit = values['-CLAHE_CLIP-']
        self.isovalue = values['-ISOVALUE-']
        self.low_color = (values['-LOW_R-'], values['-LOW_G-'], values['-LOW_B-'])
        self.high_color = (values['-HIGH_R-'], values['-HIGH_G-'], values['-HIGH_B-'])
        self.opacity = values['-OPACITY-']
        self.render_mode = "Volume" if values['-VOLUME_MODE-'] else "Mesh"
    
    def update_info_display(self, text: str):
        if hasattr(self, 'window'):
            self.window['-INFO-'].update(text)
    
    def format_volume_info(self, info: dict) -> str:
        if not info:
            return "No volume info available"
        
        size = info.get('size', 'Unknown')
        spacing = info.get('spacing', 'Unknown')
        pixel_type = info.get('pixel_type', 'Unknown')
        dimensions = info.get('dimensions', 'Unknown')
        
        return f"Size: {size}\n" \
               f"Spacing: {spacing}\n" \
               f"Pixel Type: {pixel_type}\n" \
               f"Dimensions: {dimensions}D"
    
    def format_mesh_info(self, info: dict) -> str:
        if not info:
            return "No mesh info"
        
        points = info.get('points', 0)
        cells = info.get('cells', 0)
        memory = info.get('memory_mb', 0)
        
        if points > 0:
            if points < 1000:
                complexity = "Simple"
            elif points < 5000:
                complexity = "Medium"
            else:
                complexity = "Complex"
        else:
            complexity = "Unknown"
        
        return f"Geometry:\n" \
               f"  • Points: {points:,}\n" \
               f"  • Triangles: {cells:,}\n" \
               f"  • Complexity: {complexity}\n" \
               f"  • Memory: {memory} KB\n" \
               f"Status: Ready"
    
    def load_specific_dicom(self, dicom_file, recommended_settings):
        import tempfile
        import shutil
        
        source_path = os.path.join('real_dicom', dicom_file)
        
        if not os.path.exists(source_path):
            sg.popup_error(f"DICOM file {dicom_file} not found!")
            return
        
        temp_dir = tempfile.mkdtemp(prefix='dicom_single_')
        temp_file = os.path.join(temp_dir, dicom_file)
        
        try:
            shutil.copy2(source_path, temp_file)
            
            if self.load_dicom_data(temp_dir):
                self.window['-VOLUME_MODE-'].update(True)
                self.window['-MESH_MODE-'].update(False)
                
                self.window['-GAUSSIAN_SIGMA-'].update(recommended_settings['sigma'])
                self.window['-CLAHE_CLIP-'].update(recommended_settings['clahe'])
                self.window['-ISOVALUE-'].update(recommended_settings['isovalue'])
                
                if 'opacity' in recommended_settings:
                    self.window['-OPACITY-'].update(recommended_settings['opacity'])
                else:
                    self.window['-OPACITY-'].update(0.6)
                
                self.update_slider_values()
                
                file_size = os.path.getsize(source_path) / (1024 * 1024)
                
                opacity_value = recommended_settings.get('opacity', 0.6)
                self.update_info_display(f"DICOM file loaded: {dicom_file}\n"
                                       f"Size: {file_size:.1f} MB\n\n"
                                       f"Settings applied:\n"
                                       f"  • Gaussian σ: {recommended_settings['sigma']}\n"
                                       f"  • CLAHE Clip: {recommended_settings['clahe']}\n"
                                       f"  • Isovalue: {recommended_settings['isovalue']}\n"
                                       f"  • Opacity: {opacity_value}\n\n"
                                       f"Press 'Apply' to visualize")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def load_preset_model(self, model_name):
        model_paths = {
            'bunny': 'data/bunny.obj',
            'cow': 'data/cow.obj', 
            'dragon': 'data/dragon.obj',
            'suzanne': 'data/suzanne.obj',
            'teapot': 'data/teapot.obj'
        }
        
        if model_name in model_paths:
            obj_file = model_paths[model_name]
            if os.path.exists(obj_file):
                self.window['-OBJ_FILE-'].update(obj_file)
                self.load_obj_data(obj_file)
                self.window['-MESH_MODE-'].update(True)
                self.window['-VOLUME_MODE-'].update(False)
                
                recommended_settings = {
                    'suzanne': {'sigma': 0.8, 'opacity': 0.9},
                    'bunny': {'sigma': 1.2, 'opacity': 0.8},
                    'cow': {'sigma': 1.5, 'opacity': 0.85},
                    'dragon': {'sigma': 2.0, 'opacity': 0.7},
                    'teapot': {'sigma': 1.0, 'opacity': 0.9}
                }
                
                if model_name in recommended_settings:
                    settings = recommended_settings[model_name]
                    self.window['-GAUSSIAN_SIGMA-'].update(settings['sigma'])
                    self.window['-OPACITY-'].update(settings['opacity'])
                    self.update_slider_values()
                
                self.update_info_display(f"Model loaded: {model_name.upper()}\n"
                                       f"Settings applied")
            else:
                sg.popup_error(f"File {obj_file} not found!")
    
    def update_slider_values(self, values=None):
        if hasattr(self, 'window'):
            if values is None:
                event, values = self.window.read(timeout=1)
                if values is None:
                    return
                
            self.window['-SIGMA_VAL-'].update(f"{values['-GAUSSIAN_SIGMA-']:.1f}")
            self.window['-CLAHE_VAL-'].update(f"{values['-CLAHE_CLIP-']:.1f}")
            self.window['-ISO_VAL-'].update(f"{int(values['-ISOVALUE-'])}")
            self.window['-OPACITY_VAL-'].update(f"{values['-OPACITY-']:.2f}")
    
    def run(self):
        layout = self.create_layout()
        
        self.window = sg.Window('3D Tomography Visualization', layout, 
                               finalize=True, resizable=True, icon=None)
        
        self.update_info_display("Application started!\n\n"
                               "Instructions:\n"
                               "1. Select OBJ model or DICOM data using quick buttons\n"
                               "2. For DICOM: use 'Load DICOM' button\n"
                               "3. Adjust processing parameters\n"
                               "4. Press 'Apply' for visualization\n\n"
                               "Real medical DICOM data available for testing")
        
        self.update_slider_values()
        
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WIN_CLOSED:
                break
            
            elif event == '-PRESET_BUNNY-':
                self.load_preset_model('bunny')
            elif event == '-PRESET_COW-':
                self.load_preset_model('cow')
            elif event == '-PRESET_DRAGON-':
                self.load_preset_model('dragon')
            elif event == '-PRESET_SUZANNE-':
                self.load_preset_model('suzanne')
            elif event == '-PRESET_TEAPOT-':
                self.load_preset_model('teapot')
                
            elif event == '-DICOM_BRAIN-':
                self.load_specific_dicom('brain_mri.dcm', {'sigma': 1.5, 'clahe': 2.0, 'isovalue': 50, 'opacity': 0.8})
            elif event == '-DICOM_MULTI-':
                self.load_specific_dicom('multi_layer.dcm', {'sigma': 1.0, 'clahe': 1.5, 'isovalue': 200, 'opacity': 0.6})
                
            elif event == '-LOAD_DICOM-':
                dicom_dir = values['-DICOM_DIR-']
                if dicom_dir and os.path.exists(dicom_dir):
                    self.load_dicom_data(dicom_dir)
                else:
                    sg.popup_error("Select valid DICOM folder")
                    
            elif event == '-LOAD_OBJ-':
                obj_file = values['-OBJ_FILE-']
                if obj_file and os.path.exists(obj_file):
                    self.load_obj_data(obj_file)
                else:
                    sg.popup_error("Select valid OBJ file")
                    
            elif event == '-APPLY-':
                self.update_parameters_from_gui(values)
                self.process_and_render()
                
            elif event == '-RESET-':
                self.window['-GAUSSIAN_SIGMA-'].update(1.0)
                self.window['-CLAHE_CLIP-'].update(2.0)
                self.window['-ISOVALUE-'].update(128.0)
                self.window['-LOW_R-'].update(0.0)
                self.window['-LOW_G-'].update(0.2)
                self.window['-LOW_B-'].update(0.4)
                self.window['-HIGH_R-'].update(1.0)
                self.window['-HIGH_G-'].update(0.8)
                self.window['-HIGH_B-'].update(0.6)
                self.window['-OPACITY-'].update(0.3)
                self.window['-VOLUME_MODE-'].update(True)
                self.window['-MESH_MODE-'].update(False)
                self.update_slider_values()
                
            elif event in ['-GAUSSIAN_SIGMA-', '-CLAHE_CLIP-', '-ISOVALUE-', '-OPACITY-']:
                self.update_slider_values(values)
                
        self.window.close()


if __name__ == '__main__':
    app = TomographyGUI()
    app.run() 