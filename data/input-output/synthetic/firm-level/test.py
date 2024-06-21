from synthetic_data_generator import SyntheticDataGenerator
from data_plotter import DataPlotter
import jax

key = jax.random.PRNGKey(0)

generator_ecuador = SyntheticDataGenerator('Ecuador', '2015')
synthetic_data_ecuador = generator_ecuador.generate(key, 400)

plotter_ecuador = DataPlotter(synthetic_data_ecuador)
plotter_ecuador.plot()

generator_hungary = SyntheticDataGenerator('Hungary', '2021')
synthetic_data_hungary = generator_hungary.generate(key, 400)

plotter_hungary = DataPlotter(synthetic_data_hungary)
plotter_hungary.plot()