from generators import SyntheticDataGenerator
from plotter import DataPlotter
import jax

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    generator_ecuador = SyntheticDataGenerator('Ecuador', '2015')
    synthetic_data_ecuador = generator_ecuador.generate_data(key, 4000)

    plotter_ecuador = DataPlotter(synthetic_data_ecuador)
    plotter_ecuador.save_plot('ecuador_2015_plot.png')

    generator_hungary = SyntheticDataGenerator('Hungary', '2021')
    synthetic_data_hungary = generator_hungary.generate_data(key, 4000)

    plotter_hungary = DataPlotter(synthetic_data_hungary)
    plotter_hungary.save_plot('hungary_2021_plot.png')
