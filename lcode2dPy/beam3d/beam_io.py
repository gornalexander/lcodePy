import numpy as np

from ..config.config import Config
from .data import BeamParticles, concatenate_beam_layers


class BeamSource:
    """
    This class helps to extract a beam layer from beam particles array.
    """
    # Do we really need this class?
    def __init__(self, config: Config, beam: BeamParticles):
        # From config:
        self.xi_step_size = config.getfloat('xi-step')
        
        # Get the whole beam or a beam layer:
        beam.xi_sorted()
        self.beam = beam

        # Dropped sorted_idxes = argsort(-self.beam.xi)...
        # It needs to be somewhere!

        # Shows how many particles have already deposited:
        self.layout_count = 0 # or _used_count in beam2d

    def get_beam_layer_to_layout(self, plasma_layer_idx):
        """
        Find all beam particles between plasma_layer_idx and
        plasma_layer_idx + 1, return them as a layer (class BeamParticles).
        """
        xi_min = - self.xi_step_size * plasma_layer_idx
        xi_max = - self.xi_step_size * (plasma_layer_idx + 1)

        # We use this only to speed up the search of requisite particles. Can
        # be dropped by changing to arr_to_search = self.beam.xi and not using 
        # layout_count at all.
        begin = self.layout_count
        arr_to_search = self.beam.xi[begin:]

        # Here we find the length of a layer where requisite particles lay.
        if len(arr_to_search) != 0:
            layer_length = np.sum((xi_max <= arr_to_search) *
                                  (arr_to_search < xi_min))
        else:
            layer_length = 0
        self.layout_count += layer_length

        # Here we create the array of indexes of requisite particles
        # and return the beam layer of these particles.
        indexes_arr = np.arange(begin, begin + layer_length)
        return self.beam.get_layer(indexes_arr)


class BeamDrain:
    """
    This class is used to store beam particles when the calculation of their
    movement ends.
    """
    def __init__(self):
        # We create two empty BeamParticles classes. Don't really like how it
        # is done. We need to change this procces.
        self.beam_buffer = BeamParticles(0)
        self.lost_buffer = BeamParticles(0)

    def push_beam_layer(self, beam_layer: BeamParticles):
        """
        Add a beam layer that was moved to the beam buffer.
        """
        if beam_layer.size > 0:
            self.beam_buffer = concatenate_beam_layers(self.beam_buffer,
                                                       beam_layer)

    def push_beam_lost(self, lost_layer: BeamParticles):
        """
        Add lost beam particles to the buffer of lost particles.
        """
        if lost_layer.size > 0:
            self.lost_buffer = concatenate_beam_layers(self.lost_buffer,
                                                       lost_layer)