from __future__ import absolute_import, division, print_function

import operator
import os
import re
from functools import lru_cache, reduce
from pathlib import Path

import karabo_data
import numpy as np

from scitbx.array_family import flex

from dxtbx.format.Format import Format
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.model import Scan

# RE_MODULE =re.compile(r"MODULE_(\d+)")
RE_MODULE = re.compile(r"SPB_IRDA_JNGFR/DET/MODULE_(\d+):daqOutput")


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def get_module_number(hfile):
    module_name = list(hfile["INSTRUMENT"]["SPB_IRDA_JNGFR"]["DET"].keys())[0]
    module_num = RE_MODULE.match(module_name).group(1)
    return module_num


def get_module_group(module, modules):
    return modules[module]["INSTRUMENT"]["SPB_IRDA_JNGFR"]["DET"][
        "MODULE_{}:daqOutput".format(module)
    ]["data"]


WIDTH = 1024 * 2
HEIGHT = 512 * 3
PIXEL_SIZE = 75.0 / 1000


def _get_max_pulse_count(run):
    """Looks into the H5 files in a run and extracts the maximum pulse shape"""
    pulses = []
    # breakpoint()
    for _file in run.files:
        print(_file)
        try:
            det_group = _file._file["INSTRUMENT"]["SPB_IRDA_JNGFR"]["DET"]
        except (KeyError, TypeError):
            print("  Nothing in ", _file)
            continue
        print("  Reading modules of ", det_group.keys())
        module_names = [x for x in det_group.keys() if x.startswith("MODULE")]
        for module in module_names:
            pulses.append(det_group[module]["data"]["adc"].shape[1])
    return max(pulses)


class FormatHDF5EuXFELJungfrau(FormatHDF5):
    @staticmethod
    def understand(image_file):
        # For expediency identify via path name
        image_file = Path(image_file).absolute()
        return "p002566" in str(image_file) or "EuXFEL" in str(image_file)

    def _get_module(self, module, name="data.adc"):
        # key = (module, name)
        # if not key in self._module_datasets:
        return self._run.get_virtual_dataset(
            "SPB_IRDA_JNGFR/DET/MODULE_{}:daqOutput".format(module), name
        )

    def _start(self):
        print("Jungfrau start")
        self._path = Path(os.path.dirname(self.get_image_file()))
        self._run = karabo_data.RunDirectory(str(self._path))
        # datadir = Path(sys.argv[1])
        self._modules = sorted(
            [
                RE_MODULE.match(x).group(1)
                for x in self._run.instrument_sources
                if RE_MODULE.match(x)
            ]
        )

        # Get the (dynamic) pulse structure from the file
        self._pulses = self._run.get_array(
            "SPB_RR_SYS/MDL/BUNCH_PATTERN", "sase1.nPulses.value"
        )
        # And a lookup for index from pulse
        self._pulse_index = np.cumsum(self._pulses)
        self._num_images = int(self._pulse_index[-1])

        # Get trains from train ids
        self._trains = len(self._run.train_ids)

        assert self._trains == len(
            self._pulses
        ), "Train count doesn't match pulse structure"
        # self._pulses = _get_max_pulse_count(self._run)
        print("Trains: {}, pulses: {}".format(self._trains, int(self._pulses.sum())))
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
            beam_centre=(WIDTH / 2 * PIXEL_SIZE, HEIGHT / 2 * PIXEL_SIZE),
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
        # )
        energy = self._run.get_array(
            "ACC_SYS_DOOCS/CTRL/BEAMCONDITIONS", "energy.value"
        )
        print("Ohai beam")
        return self._beam_factory.simple(0.979)  # 12398.4)

    def _goniometer(self, index=None):
        return None

    def _scan(self, index=None):
        return Scan((1, self.get_num_images()), (0, 0))

    def get_num_images(self):
        # return len(self._images)
        # return self._trains * self._pulses
        return self._num_images

    @lru_cache(maxsize=2)
    def _get_train(self, index):
        return self._run.select("*/DET/*", "data.*").train_from_index(index)

    def _get_train_pulse(self, index):
        """Get the train index and pulse number from a global index"""
        # Find where this pulse number comes. Zero-based index.
        train_index = np.searchsorted(self._pulse_index, index, side="right")
        # work out the index for the start of this train, so we can get the
        # intra-train pulse number
        train_base_pulse = 0
        if train_index > 0:
            train_base_pulse = int(self._pulse_index[train_index - 1])

        pulse = index - train_base_pulse

        return (train_index, pulse)

    def get_raw_data(self, index=0):
        # detector_2d_assembled_1 = self._run["detector_2d_assembled_1"]
        # tag = detector_2d_assembled_1[self._images[index]]
        # return flex.double(tag["detector_data"].value.astype(np.float64))

        # Construct a large image for this index
        image_data = np.zeros((HEIGHT, WIDTH))
        train, pulse = self._get_train_pulse(index)
        # # (HEIGHT, WIDTH))
        # train_index = np.searchsorted(self._pulse_index, index+1, side="right")
        # pulse = index - self._pulse_index[train_index]

        print("Reading train: {} pulse: {}".format(train, pulse))
        train_data = self._get_train(train)[1]
        # print("Assembling")
        # def _get_module(self, module, name="data.adc"):
        #     # key = (module, name)
        #     # if not key in self._module_datasets:
        #     return self._run.get_virtual_dataset(
        #         "SPB_IRDA_JNGFR/DET/MODULE_{}:daqOutput".format(module), name
        #     )
        arrange = [["8", "1"], ["7", "2"], ["6", "3"]]
        for y, mods in enumerate(arrange):
            for x, modulename in enumerate(mods):
                # Work out where to copy this module to
                target = image_data[y * 512 : (y + 1) * 512, x * 1024 : (x + 1) * 1024]

                module_data_key = "SPB_IRDA_JNGFR/DET/MODULE_{}:daqOutput".format(
                    modulename
                )
                if "data.adc" in train_data[module_data_key]:
                    module = train_data[module_data_key]["data.adc"]
                    if x == 0:
                        np.copyto(target, np.flip(module[pulse], 0))
                    else:
                        np.copyto(target, np.flip(module[pulse], 1))

                    if modulename == "6":
                        target[256:, 256:512].fill(-2)
                    if modulename == "7":
                        target[:256, 256 : 256 + 128].fill(-2)
                        pass
                else:
                    print(
                        "Warning: Module {} has no data for train {}".format(
                            modulename, train
                        )
                    )
                    target.fill(-2)
                # except:
                #     with open("Error.log", "wt") as f:
                #         mname = "SPB_IRDA_JNGFR/DET/MODULE_{}:daqOutput".format(modulename)
                #         print("ERROR: Accessing data", file=f)
                #         print("Index: {} Train: {} Pulse: {}".format(index, train, pulse), file=f)
                #         print("for train for module {}".format(modulename), file=f)
                #         print("Accessing ", mname, file=f)
                #         print("On train data", train_data, file=f)
                #         print("  train_data Keys: ", train_data.keys(), file=f)
                #         if mname in train_data:
                #             mod = train_data[mname]
                #             print("Module: ", mod, file=f)
                #             print("   Module Keys: ", mod.keys(), file=f)
                #         else:
                #             print(mname, "NOT IN TRAINDATA", file=f)
                #         # printtrain_data[
                #         #     "SPB_IRDA_JNGFR/DET/MODULE_{}:daqOutput".format(modulename)
                #         # ]()
                #         # print(train_data)

                #         raise
                #                     # print("    ", module.shape)
                # subset = module[train, pulse]

                target[0].fill(-2)
                target[-1].fill(-2)
                target[:, 0].fill(-2)
                target[:, -1].fill(-2)
                target[255:257].fill(-2)
                target[:, 255:257].fill(-2)
                target[:, 511:513].fill(-2)
                target[:, 767:769].fill(-2)

                # module = image_data[y * 512 : (y + 1) * 512, x * 1024 : (x + 1) * 1024]
        # Do masking
        # image_data[0].fill(-2)
        # # image_data[]
        # image_data[-1].fill(-2)
        # Do some basic scaling
        image_data[image_data > 0] *= 0.01

        return flex.double(image_data.astype(np.float64))

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_detector(self, index=None):
        return self._detector_instance

    def get_beam(self, index=None):
        return self._beam_instance
