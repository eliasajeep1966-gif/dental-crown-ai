import os
import torch
import trimesh
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points
import h5py

def random_rotate_point_cloud(pna,crown,normals, rotation_axes=('x', 'y', 'z')):
    rotation_matrix = torch.eye(3)
    
    if 'x' in rotation_axes:
        angle_x = np.random.uniform(0, 2 * np.pi)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        rot_x = torch.tensor([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], dtype=torch.float32)
        rotation_matrix = rot_x @ rotation_matrix
    
    if 'y' in rotation_axes:
        angle_y = np.random.uniform(0, 2 * np.pi)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        rot_y = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32)
        rotation_matrix = rot_y @ rotation_matrix

    if 'z' in rotation_axes:
        angle_z = np.random.uniform(0, 2 * np.pi)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        rot_z = torch.tensor([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        rotation_matrix = rot_z @ rotation_matrix

    rotated_pna = torch.matmul(pna, rotation_matrix)
    rotated_crown = torch.matmul(crown,rotation_matrix)
    rotated_crown_normals = torch.matmul(normals,rotation_matrix)
    return rotated_pna,rotated_crown,rotated_crown_normals

class IOS_Dataset(Dataset):
    def __init__(self, root_dir, is_train=True, crop_size=(20.0, 20.0, 20.0), sample_points=1024, template_dir=None):
        self.istrain = is_train
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.sample_points = sample_points
        self.template_dir = (
            Path(template_dir).expanduser().resolve()
            if template_dir is not None
            else Path(__file__).resolve().parents[1] / 'fdi_template'
        )
        if not self.template_dir.is_dir():
            raise FileNotFoundError(f'FDI template directory not found: {self.template_dir}')
        self.subdirs = ['14','15','16', '17', '24', '25', '26', '27','34', '36', '37','44', '45', '46', '47' ]
       
       
        self.data_paths = []
        for subdir in self.subdirs:
            subdir_path = self.root_dir / subdir
            if is_train:
                case_dir = subdir_path / 'train' 
            else:
                case_dir = subdir_path / 'test'
            if not case_dir.is_dir():
                continue
            for case_path in sorted(case_dir.iterdir()):
                if not case_path.is_dir():
                    continue
                abs_case = str(case_path)
                crown_file = os.path.join(abs_case, 'crown.ply')
                pna_crop_file = os.path.join(abs_case, 'pna_crop.ply')
                psr_grid = os.path.join(abs_case,'psr_my.npz')
                curvature = os.path.join(abs_case,'crown_attributes.h5')
                prepare_margin = os.path.join(abs_case,'boundary_labels.npy')
                self.data_paths.append((int(subdir), abs_case, crown_file,pna_crop_file,psr_grid,curvature,prepare_margin))

        if not self.data_paths:
            split = 'train' if is_train else 'test'
            raise FileNotFoundError(f'No case folders found under {self.root_dir}/<tooth>/{split}')

    def __len__(self):
        return len(self.data_paths)
    
    def normalize_point_cloud(self, point_cloud, transition=0.0, scale=1.0, center=None):
        """Normalize a cloud; pass one shared center for aligned crown/PNA inputs."""
        if center is None:
            center = (torch.min(point_cloud, dim=0)[0] + torch.max(point_cloud, dim=0)[0]) / 2
        else:
            center = torch.as_tensor(center, dtype=point_cloud.dtype, device=point_cloud.device)

        crop_scale = torch.as_tensor(self.crop_size, dtype=point_cloud.dtype, device=point_cloud.device) / 2
        return (point_cloud - center) / crop_scale
    
    def get_curvature(self,dirpath):
        with h5py.File(dirpath, 'r') as f:
            curvatures = np.asarray(f['curvatures'][:], dtype=np.float32)
            margine = np.asarray(f['whetherismarginline'][:], dtype=np.float32)

        if curvatures.ndim != 1 or curvatures.size == 0:
            raise ValueError('curvatures must be a non-empty 1-D array')
        if margine.shape != curvatures.shape:
            raise ValueError('curvatures and whetherismarginline must have matching shapes')
        if not np.isfinite(curvatures).all() or not np.isfinite(margine).all():
            raise ValueError('curvatures and whetherismarginline must contain only finite values')

        curv_min = float(np.min(curvatures))
        curv_range = float(np.max(curvatures)) - curv_min
        if curv_range <= np.finfo(np.float32).eps:
            normalized_curvatures = np.zeros_like(curvatures)
        else:
            # The upstream expression omitted parentheses and did not perform min-max scaling.
            normalized_curvatures = (curvatures - curv_min) / curv_range

        curvatures_tensor = torch.as_tensor(normalized_curvatures, dtype=torch.float32)
        margine_tensor = torch.as_tensor(margine, dtype=torch.float32)
        return curvatures_tensor, margine_tensor

    def crop_mesh(self, mesh: trimesh.Trimesh, center: np.ndarray, size: float,has_pre_pare=False):
        vertices = mesh.vertices

        offset = size[0] / 2
        in_box = np.all(np.abs(vertices - center) <= offset, axis=1)

        cropped_vertices = vertices[in_box]
        cropped_faces = mesh.faces[np.all(in_box[mesh.faces], axis=1)]
      
        new_mesh = trimesh.Trimesh(vertices=cropped_vertices, faces=cropped_faces, process=False)
        if has_pre_pare:
            cropped_attributes = {
                name: attr[in_box] for name, attr in mesh.vertex_attributes.items()
            }

            new_mesh.vertex_attributes.update(cropped_attributes)

        return new_mesh    
    
    def get_hemisphere_template(self):
        radius = 7.5
        
        theta = torch.acos(2 * torch.rand(self.sample_points) - 1)
        phi = torch.rand(self.sample_points) * 2 * np.pi

        x = radius * torch.sin(theta) * torch.cos(phi)
        y = radius * torch.sin(theta) * torch.sin(phi)
        z = radius * torch.cos(theta)

        hemisphere_points = torch.stack((x, y, z), dim=1)

        hemisphere_points = self.normalize_point_cloud(hemisphere_points)
        
        return hemisphere_points

    def get_template(self, tooth_number):
        """Load the FDI template from the bundled or explicitly configured directory."""
        template_path = self.template_dir / f'{int(tooth_number)}crown.ply'
        if not template_path.is_file():
            raise FileNotFoundError(f'FDI template not found for tooth {tooth_number}: {template_path}')
        template_mesh = trimesh.load(template_path, process=False)
        if isinstance(template_mesh, trimesh.Scene):
            template_mesh = template_mesh.dump(concatenate=True)
        if not isinstance(template_mesh, trimesh.Trimesh) or len(template_mesh.vertices) == 0:
            raise ValueError(f'FDI template is empty or invalid: {template_path}')
        template_vertices = template_mesh.vertices
        template_vertices = torch.tensor(template_vertices,dtype=torch.float32)
        template_vertices = self.normalize_point_cloud(template_vertices)
        template_vertices = sample_farthest_points(template_vertices[None,...],K=self.sample_points)[0].squeeze()
        return template_vertices

    def __getitem__(self, idx):
        tooth_number, dirpath, crown_file, pna_crop_file,psr_file,attributes,prepared_tooth_npy = self.data_paths[idx]
     
        crown_mesh = trimesh.load(crown_file,process=False)
        prepare_margin_label = np.load(prepared_tooth_npy)
       
        crown_vertices = crown_mesh.vertices
        pna_crop_vertices = trimesh.load(pna_crop_file,process=False)
        pna_crop_vertices.vertex_attributes['margin'] = prepare_margin_label
        crown_min_bound, crown_max_bound = crown_vertices.min(axis=0), crown_vertices.max(axis=0)
        crown_center = (crown_min_bound + crown_max_bound) / 2
        
        crown_cropped_mesh = self.crop_mesh(crown_mesh, crown_center, self.crop_size)
        pna_crop_cropped_mesh = self.crop_mesh(pna_crop_vertices, crown_center, self.crop_size,True) 
        cropped_margin_label = pna_crop_cropped_mesh.vertex_attributes['margin']
        cropped_margin_label = torch.tensor(cropped_margin_label[:, None], dtype=torch.float32)
        crown_tensor = torch.tensor(crown_cropped_mesh.vertices, dtype=torch.float32)
        crown_normal_tensor = torch.tensor(crown_cropped_mesh.vertex_normals, dtype=torch.float32)
        pna_crop_tensor = torch.tensor(pna_crop_cropped_mesh.vertices, dtype=torch.float32)
        psr_grid = torch.tensor(np.load(psr_file)['psr'],dtype=torch.float32)
        template = self.get_template(tooth_number)
        
        if self.istrain and False :
            pna_crop_tensor,crown_tensor,crown_normal_tensor = random_rotate_point_cloud(pna_crop_tensor,crown_tensor,crown_normal_tensor, rotation_axes=('x', 'y', 'z'))
            scale = np.random.uniform(0.9, 1.1)
            translation = np.random.uniform(-0.1, 0.1, size=(3,))
            translation_tensor = torch.tensor(translation, dtype=torch.float32)
            crown_tensor = crown_tensor * scale + translation_tensor
            pna_crop_tensor = pna_crop_tensor * scale + translation_tensor

        # Use the same origin and scale so the crown remains aligned to the jaw scan.
        crown_tensor = self.normalize_point_cloud(crown_tensor, center=crown_center)
        pna_crop_tensor = self.normalize_point_cloud(pna_crop_tensor, center=crown_center)
       
        crown_sampled, crown_sample_idx = sample_farthest_points(crown_tensor[None, :, :], K=4*self.sample_points)
        crown_sampled = crown_sampled.squeeze()
        curvatures,margine = self.get_curvature(attributes)
        crown_curvatures = curvatures[crown_sample_idx[0]]
        margine = margine[crown_sample_idx[0]]
        pna_crop_sampled,pna_sample_idx = sample_farthest_points(pna_crop_tensor[None, :, :], K=2*self.sample_points)
        cropped_margin_label = cropped_margin_label[pna_sample_idx[0]]
        pna_crop_sampled = torch.cat([pna_crop_sampled.squeeze(),cropped_margin_label],dim=1)
        
        return pna_crop_sampled, crown_sampled, crown_center, template,psr_grid, crown_curvatures,margine,dirpath
