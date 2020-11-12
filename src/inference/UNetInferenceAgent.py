"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        volume = med_reshape(volume, new_shape=(volume.shape[0], self.patch_size, self.patch_size))
        volume = self.single_volume_inference(volume)
        
        return volume

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # Put all slices into a 3D Numpy array.
        
        def inference(img):
            img = torch.from_numpy(img.astype(np.single) / np.max(img)).unsqueeze(0).unsqueeze(0)
            pred = self.model(img.to(self.device))
            return np.squeeze(pred.cpu().detach())
        
        for idx in range(volume.shape[0]):
            pred = inference(volume[idx, :, :])
            slices.append(torch.argmax(pred, dim=0).numpy())
        
        slices = np.array([slc for slc in slices])

        return slices
