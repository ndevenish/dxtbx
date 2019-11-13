from __future__ import absolute_import, division, print_function

import os
import sys

import h5py
import numpy as np

import karabo_data

from scitbx import matrix
from scitbx.array_family import flex

from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.model import Scan

import re
import operator
from pathlib import Path

from functools import reduce


# RE_MODULE =re.compile(r"MODULE_(\d+)")
RE_MODULE = re.compile(r"SPB_IRDA_JNGFR/DET/MODULE_(\d+):daqOutput")

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def get_module_number(hfile):
    module_name = list(hfile["INSTRUMENT"]["SPB_IRDA_JNGFR"]["DET"].keys())[0]
    module_num = RE_MODULE.match(module_name).group(1)
    return module_num

def get_module_group(module, modules):
    return modules[module]["INSTRUMENT"]["SPB_IRDA_JNGFR"]["DET"]["MODULE_{}:daqOutput".format(module)]["data"]

WIDTH = 1024*2
HEIGHT = 512 * 3
PIXEL_SIZE = 75.0 / 1000

class FormatHDF5EuXFELJungfrau(FormatHDF5):
    @staticmethod
    def understand(image_file):
        # For expediency identify via path name
        return "p002566" in image_file

    def _get_module(self, module, name="data.adc"):
        # key = (module, name)
        # if not key in self._module_datasets:
        return self._run.get_virtual_dataset("SPB_IRDA_JNGFR/DET/MODULE_{}:daqOutput".format(module),name)

    def _start(self):
        print("Jungfrau start")
        self._path = Path(os.path.dirname(self.get_image_file()))
        self._run = karabo_data.RunDirectory(str(self._path))
        # datadir = Path(sys.argv[1])
        self._modules = [RE_MODULE.match(x).group(1) for x in self._run.instrument_sources if RE_MODULE.match(x)]
        # self._module_datasets = {}
        # Open one module to get shape of train/pulses
        module = self._get_module(self._modules[0], "data.adc")
        self._trains, self._pulses = module.shape[:2]

        # breakpoint()
        # energies = self._run.get_array("ACC_SYS_DOOCS/CTRL/BEAMCONDITIONS", "energy.value")
        
        # filenames = list(self._path.glob("*-JNGFR*.h5"))
        # # Open all files and get module
        # self._files = [h5py.File(x) for x in filenames]
        # self._modules = {get_module_number(x): x for x in files}
        # # Get the shape/train count
        # a_module = list(modules.keys())[0]
        # a_module_shape = get_module_group(a_module, modules)["adc"].shape
        #  = a_module_shape[0]
        # self._pulses = a_module_shape[1]
        # image_count = prod(a_module_shape[:2])
        # print("Images: {} in {} trains and {} pulses".format(image_count, self._trains, self._pulses))
        # breakpoint()
    # prod(g["adc"].shape[:2])

        # self._run = karabo_data.RunDirectory()
        # self._run.info()
        # breakpoint()
        # self._h5_handle = h5py.File(self.get_image_file(), "r")
        # self._run = FormatHDF5Sacla._get_run_h5group(self._h5_handle)
        # event_info = self._run["event_info"]
        # tag_number_list = event_info["tag_number_list"]
        # self._images = ["tag_%d" % tag for tag in tag_number_list]


    def _detector(self, index=None):
#        # Get the pixel and image size
#        detector_2d_assembled_1 = self._run["detector_2d_assembled_1"]
#        detector_info = detector_2d_assembled_1["detector_info"]
#        pixel_size = (
#            detector_info["pixel_size_in_micro_meter"][0] / 1000,
#            detector_info["pixel_size_in_micro_meter"][1] / 1000,
#        )
#        tag = detector_2d_assembled_1[self._images[0]]
#        data = tag["detector_data"][()]
#
#        # detector_image_size is fast-, slow- , just in case the dataset is ever non-square
#        image_size = (data.shape[1], data.shape[0])
#        trusted_range = (0, 200000)
#
#        # Initialise detector frame
#        fast = matrix.col((1.0, 0.0, 0.0))
#        slow = matrix.col((0.0, -1.0, 0.0))
#        orig = matrix.col(
#            (
#                -image_size[0] * pixel_size[0] / 2,
#                image_size[1] * pixel_size[1] / 2,
#                -100.0,
#            )
#        )
#        # Make the detector
#        return self._detector_factory.make_detector(
#            "", fast, slow, orig, pixel_size, image_size, trusted_range
#        )
        return self._detector_factory.simple(
                sensor="PAD",
                distance=260,
                beam_centre=(
                    WIDTH / 2 * PIXEL_SIZE,
                    HEIGHT / 2 * PIXEL_SIZE,
                ),
                fast_direction="+x",
                slow_direction="+y",
                pixel_size=(PIXEL_SIZE, PIXEL_SIZE),
                image_size=(WIDTH, HEIGHT),
                trusted_range=(-1, 65535),
                mask=[],
            )
        print("Jung _detector")
        pass

    def _beam(self, index=None):
 #       run_info = self._run["run_info"]
 #       sacla_config = run_info["sacla_config"]
 #       eV = sacla_config["photon_energy_in_eV"].value
#
#        return self._beam_factory.simple(12398.4 / eV)
        # print("Jung _beam"
        #)
        energy = self._run.get_array("ACC_SYS_DOOCS/CTRL/BEAMCONDITIONS", "energy.value")
        return self._beam_factory.simple(12398.4)

    def _goniometer(self, index=None):
    	return None
    # def _scan(self, index=None):
    #     return None

    def _scan(self, index=None):
        return Scan((1,self._trains*self._pulses+1), (0,0), 0)
    def get_num_images(self):
        #return len(self._images)
        return self._trains * self._pulses

    def get_raw_data(self, index=0):
        #detector_2d_assembled_1 = self._run["detector_2d_assembled_1"]
        #tag = detector_2d_assembled_1[self._images[index]]
        #return flex.double(tag["detector_data"].value.astype(np.float64))
        pass

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_detector(self, index=None):
        return self._detector_instance

    def get_beam(self, index=None):
        return self._beam_instance

