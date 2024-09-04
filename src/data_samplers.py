import torch
import numpy as np

class Sampler:
    def __init__(self, device='cuda'):
        self.device = device
    
    def sample(self, size=5):
        raise NotImplementedError("The sample method must be overridden by subclasses")

class TorusSampler(Sampler):
    def __init__(self, major_radius=1.0, minor_radius=0.5, device='cuda'):
        super(TorusSampler, self).__init__(device=device)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
    
    def sample(self, batch_size=10):
        theta = torch.rand(batch_size, device=self.device) * 2 * np.pi  # Angle around the major circle
        phi = torch.rand(batch_size, device=self.device) * 2 * np.pi    # Angle around the minor circle
        
        x = (self.major_radius + self.minor_radius * torch.cos(phi)) * torch.cos(theta)
        y = (self.major_radius + self.minor_radius * torch.cos(phi)) * torch.sin(theta)
        z = self.minor_radius * torch.sin(phi)
        
        return torch.stack((x, y, z), dim=1)


class LinkedTorusSampler(Sampler):
    def __init__(self, major_radius=1.0, device='cuda'):
        super(LinkedTorusSampler, self).__init__(device=device)
        self.major_radius = major_radius
        self.minor_radius = major_radius / 5

    def sample(self, batch_size=10):
        half_batch = batch_size // 2
        
        # First torus (in XY-plane, centered on the Z-axis)
        theta1 = torch.rand(half_batch, device=self.device) * 2 * np.pi
        phi1 = torch.rand(half_batch, device=self.device) * 2 * np.pi
        x1 = (self.major_radius + self.minor_radius * torch.cos(phi1)) * torch.cos(theta1)
        y1 = (self.major_radius + self.minor_radius * torch.cos(phi1)) * torch.sin(theta1)
        z1 = self.minor_radius * torch.sin(phi1)
        torus1 = torch.stack((x1, y1, z1), dim=1)
        
        # Second torus (in YZ-plane, centered on the X-axis, shifted along Y-axis)
        theta2 = torch.rand(half_batch, device=self.device) * 2 * np.pi
        phi2 = torch.rand(half_batch, device=self.device) * 2 * np.pi
        x2 = self.minor_radius * torch.sin(phi2)
        y2 = (self.major_radius + self.minor_radius * torch.cos(phi2)) * torch.cos(theta2)
        z2 = (self.major_radius + self.minor_radius * torch.cos(phi2)) * torch.sin(theta2)
        
        # Shift the second torus so that it passes through the origin (0, 0, 0)
        y2 = y2 - self.major_radius  # Shift along the Y-axis by the major radius
        torus2 = torch.stack((x2, y2, z2), dim=1)
        
        # Concatenate the points from both tori
        linked_tori = torch.cat([torus1, torus2], dim=0)
        
        # Shuffle to mix points from both tori
        indices = torch.randperm(linked_tori.size(0), device=self.device)
        return linked_tori[indices]


import torch
import numpy as np

class TorusKnotSampler(Sampler):
    def __init__(self, p=2, q=3, major_radius=1.0, minor_radius=0.2, tube_radius=0.05, device='cuda'):
        """
        Initializes the Torus Knot Sampler.

        Parameters:
        - p: Integer, the number of times the knot wraps around the torus's central axis.
        - q: Integer, the number of times the knot passes through the hole of the torus.
        - major_radius: Float, the major radius of the torus.
        - minor_radius: Float, the minor radius of the torus.
        - tube_radius: Float, the radius of the tube (small torus around the knot).
        - device: Device to use (e.g., 'cuda' or 'cpu').
        """
        super(TorusKnotSampler, self).__init__(device=device)
        self.p = p
        self.q = q
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.tube_radius = tube_radius
        self.device = device

    def sample(self, batch_size=10):
        """
        Samples points uniformly from the surface of a small torus shaped like a torus knot.

        Parameters:
        - batch_size: Number of points to sample.

        Returns:
        - Tensor of shape (batch_size, 3) containing the sampled points.
        """
        # Uniformly sample angles for the knot
        t = torch.rand(batch_size, device=self.device) * 2 * np.pi
        theta = torch.rand(batch_size, device=self.device) * 2 * np.pi
        
        # Parametric equations for the torus knot centerline
        x_knot = (self.major_radius + self.minor_radius * torch.cos(self.q * t)) * torch.cos(self.p * t)
        y_knot = (self.major_radius + self.minor_radius * torch.cos(self.q * t)) * torch.sin(self.p * t)
        z_knot = self.minor_radius * torch.sin(self.q * t)
        
        # Calculate the derivatives (tangent vectors)
        dx_dt = -self.p * torch.sin(self.p * t) * (self.major_radius + self.minor_radius * torch.cos(self.q * t)) - \
                self.q * self.minor_radius * torch.sin(self.q * t) * torch.cos(self.p * t)
        dy_dt = self.p * torch.cos(self.p * t) * (self.major_radius + self.minor_radius * torch.cos(self.q * t)) - \
                self.q * self.minor_radius * torch.sin(self.q * t) * torch.sin(self.p * t)
        dz_dt = self.q * self.minor_radius * torch.cos(self.q * t)
        
        tangent = torch.stack((dx_dt, dy_dt, dz_dt), dim=1)
        tangent = tangent / torch.norm(tangent, dim=1, keepdim=True)
        
        # Generate an arbitrary normal vector that is not parallel to the tangent
        arbitrary_vector = torch.tensor([1, 0, 0], device=self.device).expand_as(tangent)
        normal = torch.cross(tangent, arbitrary_vector, dim=1)
        normal = normal / torch.norm(normal, dim=1, keepdim=True)
        
        # Generate the binormal vector
        binormal = torch.cross(tangent, normal, dim=1)
        
        # Parametrize the small torus around the knot
        r = self.tube_radius
        x = x_knot + r * (torch.cos(theta) * normal[:, 0] + torch.sin(theta) * binormal[:, 0])
        y = y_knot + r * (torch.cos(theta) * normal[:, 1] + torch.sin(theta) * binormal[:, 1])
        z = z_knot + r * (torch.cos(theta) * normal[:, 2] + torch.sin(theta) * binormal[:, 2])
        
        return torch.stack((x, y, z), dim=1)



class TetrahedronGaussianSampler(Sampler):
    def __init__(self, distance=1.0, std_factor=0.15, device='cuda'):
        super(TetrahedronGaussianSampler, self).__init__(device=device)
        self.distance = distance
        self.std = std_factor * distance
        # Vertices of a regular tetrahedron centered at the origin
        self.centers = torch.tensor([
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1]
        ], dtype=torch.float32, device=self.device) * (distance / np.sqrt(3))

    def sample(self, batch_size=10):
        # Randomly choose one of the four centers for each sample
        indices = torch.randint(0, 4, (batch_size,), device=self.device)
        centers = self.centers[indices]
        
        # Add Gaussian noise centered at the selected vertices
        samples = centers + torch.randn(batch_size, 3, device=self.device) * self.std
        return samples
