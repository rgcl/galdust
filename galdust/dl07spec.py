from glob import glob
from os import path
from io import BytesIO
import re
import numpy as np
from collections import namedtuple
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt

PATTERN = re.compile('^U(?P<umin>[^_]+)_(?P<umax>[^_]+)_(?P<model>.+)\\.txt$')

# this lists come from the table in the DL02spec/README.txt
# about "Spectra are given for 11 dust models"
MODEL_TABLE_MODEL = [
    'MW3.1_00',
    'MW3.1_10',
    'MW3.1_20',
    'MW3.1_30',
    'MW3.1_40',
    'MW3.1_50',
    'MW3.1_60',
    'LMC2_00',
    'LMC2_05',
    'LMC2_10',
    'smc'
]
MODEL_TABLE_Q_PATH = [
    0.47,
    1.12,
    1.77,
    2.50,
    3.19,
    3.90,
    4.58,
    0.75,
    1.49,
    2.37,
    0.10
]

ModelDataItem = namedtuple('ModelDataItem', 'umin,umax,q_pah,model,data')


class DL07specContainer:
    """

    """

    def __init__(self):
        self.loaded = False
        self.bad_model_files = []

    def _lazy_loader(self):
        if self.loaded:
            return
        dl2007spec_dir = path.join(path.dirname(__file__), 'data', 'DL07spec', '**/*.txt')
        self.models_data = dict()
        for absolute_path in glob(dl2007spec_dir):
            with open(absolute_path, 'r') as file:
                filename = absolute_path.split(path.sep)[-1]
                file_lines = file.readlines()
                if len(file_lines) < 1001:
                    self.bad_model_files.append(filename)
                    continue
                # lines inspired in pcigale 2018.0 database_builder/__init__.py:503,525
                file_content = ''.join(file_lines[-1001:])
                data = np.genfromtxt(BytesIO(file_content.encode()))
                file_info = PATTERN.match(filename)

                umin = float(file_info['umin'])
                umax = float(file_info['umax'])
                model = file_info['model']
                q_pah = DL07specContainer.model_to_q_pah(model)
                key = self.get_key(umin, umax, q_pah)

                self.models_data[key] = ModelDataItem(umin, umax, q_pah, model, data)
        self.loaded = True

    @staticmethod
    def get_key(umin, umax, q_pah):
        return f'{umin}_{umax}_{q_pah}'

    @staticmethod
    def q_pah_to_model(q_pah):
        if q_pah in MODEL_TABLE_Q_PATH:
            index = MODEL_TABLE_Q_PATH.index(q_pah)
            return MODEL_TABLE_MODEL[index]
        else:
            raise Exception(f'Model is undefined for q_pah={q_pah}')

    @staticmethod
    def model_to_q_pah(model):
        if model in MODEL_TABLE_MODEL:
            index = MODEL_TABLE_MODEL.index(model)
            return MODEL_TABLE_Q_PATH[index]
        else:
            raise Exception(f'q_pah is undefined for model={model}')

    def create_model(self, umin, umax, q_pah, gamma):
        self._lazy_loader()
        try:
            model_data = self.models_data[self.get_key(umin, umax, q_pah)]
        except KeyError:
            raise Exception(f'The model umin={umin}, umax={umax}, q_pah={q_pah} does not exists. '
                            f'Please review the .bad_model_files attribute')
        return DustModel(*model_data, gamma)

    def create_model_batch(self, gamma, model=None, umin=None, umax=None, q_pah=None):
        """
        Return a list of instances of DustModel that match the parameters
        :param gamma:
        :param model: optional
        :param umin: optional
        :param umax: optional
        :param q_pah: optional
        :return:
        """
        self._lazy_loader()
        sublist = []
        for i in self:
            model_data_item = self[i]

            # we continue to the next iteration unless model_data_item match the parameters
            if (model is not None and model_data_item.model != model) or model is None:
                continue
            if (umin is not None and model_data_item.umin != umin) or model is None:
                continue
            if (umax is not None and model_data_item.umax != umax) or model is None:
                continue
            if (q_pah is not None and model_data_item.q_pah != q_pah) or model is None:
                continue

            sublist.append(DustModel(*model_data_item, gamma))

        return np.array(sublist)

    def __len__(self):
        self._lazy_loader()
        return len(self.models_data)

    def __getitem__(self, key):
        self._lazy_loader()
        return self.models_data[key]

    def __setitem__(self, key, value):
        raise Exception('DL07spec cannot be setted in runtime')

    def __delitem__(self, key):
        raise Exception('DL07spec cannot be setted in runtime')

    def __missing__(self, key):
        raise KeyError(f'Not found row at <id={key}>')

    def __iter__(self):
        self._lazy_loader()
        self._iter = iter(self.models_data)
        return self

    def __next__(self):
        self._lazy_loader()
        # In this case, StopIteration exception is raised by the dict behind self._index
        return next(self._iter)

    def __str__(self):
        self._lazy_loader()
        return f'<DL07spec {len(self)} items>'


class DustModel:
    def __init__(self, umin, umax, q_pah, model, data, gamma):
        self._umin = umin
        self._umax = umax
        self._q_pah = q_pah
        self._model = model
        self._data = data
        self._gamma = gamma
        # lambdas in um to nm
        self._wavelength = (data[:, 0] * u.micron).to(u.nm)

        h_mass_kg = 1.67e-27  # taken from literature

        # \nu*dP/d\nu in (erg s-1 H-1) to (W/kg of H)
        nu_dp_dnu_aux = (data[:, 1] * 1/h_mass_kg * u.Unit('erg/(s*kg)')).to('W/kg')

        # using \nu*L_\nu = \lambda*L_\lambda, then L_\lambda = nu_dP_dnu_aux / \lambda
        self._luminosity = nu_dp_dnu_aux/self._wavelength

        self._j_nu = data[:, 2]  # j_\nu in (Jy cm2 sr-1 H-1)

    def spectrum(self):
        return np.column_stack([self._wavelength.value, self._luminosity.value])

    def calc_bolumetric_luminosity(self):
        """
        the sum of all magnitudes in all wavelengths
        :return:
        """
        spectrum = self.spectrum()
        return np.trapz(spectrum[:, 1], spectrum[:, 0])

    def plot_spectrum(self, filename=None):
        """
        Plot the spectrum
        :param filename:
        :return:
        """
        spectrum = self.spectrum()
        plt.plot(spectrum[:, 0], spectrum[:, 1])
        plt.title('Model Spectrum')
        plt.xlabel('$\lambda[nm]$')
        plt.ylabel('$L_\lambda$ [W/nm/(kg of H)]')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    @property
    def umin(self):
        return self._umin

    @property
    def umax(self):
        return self._umax

    @property
    def q_pah(self):
        return self._q_pah

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if 0 <= gamma <= 1:
            raise Exception('gamma must be between 0 and 1')
        self._gamma = gamma

    def __str__(self):
        return f'<dl07spec.DustModel model={self._model}, umin={self._umin}, umax={self._umax}>'
