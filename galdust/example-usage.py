#!/usr/bin/env python
import galdust

# in this case we return the container for the DL07spec models
container = galdust.dl07

# we can find the q_pah of some model name
q_pah = galdust.dl07.model_to_q_pah('LMC2_00')

# for create a model, we call the create_model mathod. The loading is lazy, so
# the first call take a while. As was required, the paremeters umin, umax, q_pah and gamma are needed.
model = container.create_model(umin=0.3, umax=1e3, q_pah=q_pah, gamma=0.1)

# we can also query for a range of models, for instance all the models.
# the first parameter is gamma, and must be provider ever, the other parameters are optionals
models = container.create_model_batch(gamma=0.5)

print(f'Batch operation return dimension {models.shape}')

# ...or only models that match cryteria
models = container.create_model_batch(gamma=0.5, model='LMC2_00', umin=0.3)

print(f'Batch operation with model=LMC2_00 and umin=0.3 return dimension {models.shape}')

# each model can return the requested properties, as example
print(f'wavelengths of the model {model} is {model.wavelength}')

# for instance the spectrum
print(model.spectrum())

# or bolumetric
print(f'bolumetric_luminosity: {model.calc_bolumetric_luminosity()}')

# or the plot
model.plot_spectrum()
