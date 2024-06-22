from NetworkVisualizer import NetworkVisualizer
import pandas as pd
if __name__ == "__main__":
    # Example usage with CSV:
    #print csv as dataframe
    df = pd.read_csv('/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/2016-2020_SML/preprocessed/SML_2016/intermediate_use_all_countries.csv')
    print(df)
    visualizer = NetworkVisualizer(csv_path='/users/mf23yyt/Thesis/network_reconstruction/data/input-output/SML/2016-2020_SML/preprocessed/SML_2016/intermediate_use_all_countries.csv', directed=True, weighted=True)
    visualizer.show_network()
    visualizer.show_heatmap()
    visualizer.show_distributions()
    visualizer.save_distributions(output_dir='test_outputs/SML_2020/intermediate_use_all_countries')