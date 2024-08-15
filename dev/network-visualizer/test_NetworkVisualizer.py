from NetworkVisualizer import NetworkVisualizer
import pandas as pd
import os 
if __name__ == "__main__":
    # Example usage with CSV:
    #print csv as dataframe
    
    #All countries 2016
    df = pd.read_csv('/users/mf23yyt/Thesis/networks/data/input-output/SML/2016-2020_SML/preprocessed/2016_SML/intermediate_use.csv')
    df.set_index('V1', inplace=True)
    index_labels = df.index.tolist()
    adjacency_matrix = df.reindex(index=index_labels, columns=index_labels)
    # print("index labels:",index_labels)
    print(f"shape: {df.shape}")
    # print(df.head())
    #shift 
    visualizer = NetworkVisualizer(adjacency_matrix = adjacency_matrix, directed=True, weighted=True)
    output_dir='test_outputs/SML_2020/countries_intermediate_use/ALL_COUNTRIES_2016'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    visualizer.save_network(output_dir=output_dir, title = "All Countries 2016", node_size=10, edge_width=0.1)
    visualizer.save_heatmap(output_dir=output_dir, title = "All Countries 2016")
    visualizer.save_distributions(output_dir=output_dir, title = "All Countries 2016", loglog= True)
    #Austria 2016
    
    # df = pd.read_csv('/users/mf23yyt/Thesis/networks/data/input-output/SML/2016-2020_SML/preprocessed/2016_SML/countries_intermediate_use/AUT_intermediate_use.csv')
    # df.set_index('V1', inplace=True)
    # index_labels = df.index.tolist()
    # adjacency_matrix = df.reindex(index=index_labels, columns=index_labels)
    # # print("index labels:",index_labels)
    # print(f"shape: {df.shape}")
    # # print(df.head())
    # #shift 
    # visualizer = NetworkVisualizer(adjacency_matrix = adjacency_matrix, directed=True, weighted=True)
    # output_dir='test_outputs/SML_2020/countries_intermediate_use/AUT_2016'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # visualizer.save_network(output_dir=output_dir, title = "Austria 2016", node_size=10, edge_width=0.1)
    # visualizer.save_heatmap(output_dir=output_dir, title = "Austria 2016")
    # visualizer.save_distributions(output_dir=output_dir, title = "Austria 2016", loglog= True)
    
    # df = pd.read_csv('/users/mf23yyt/Thesis/networks/data/input-output/SML/2016-2020_SML/preprocessed/2016_SML/countries_intermediate_use/USA_intermediate_use.csv')
    # df.set_index('V1', inplace=True)
    # index_labels = df.index.tolist()
    # adjacency_matrix = df.reindex(index=index_labels, columns=index_labels)
    # # print("index labels:",index_labels)
    # print(f"shape: {df.shape}")
    # # print(df.head())
    # #shift 
    # visualizer = NetworkVisualizer(adjacency_matrix = adjacency_matrix, directed=True, weighted=True)
    # output_dir='test_outputs/SML_2020/countries_intermediate_use/USA_2016'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # visualizer.save_network(output_dir=output_dir, title = "USA 2016", node_size=10, edge_width=0.1)
    # visualizer.save_heatmap(output_dir=output_dir, title = "USA 2016")
    # visualizer.save_distributions(output_dir=output_dir, title = "USA 2016", loglog= True)