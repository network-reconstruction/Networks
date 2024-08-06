import numpy as np
import matplotlib.pyplot as plt
import json
import time

class DegSeqAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.in_degrees = []
        self.out_degrees = []
        self.read_file()
        self.calculate_metrics()
        self.save_results()
        self.plot_degree_distribution()
        self.plot_ccdf()
        self.plot_in_vs_out_degree()

    def read_file(self):
        with open(self.filename, 'r') as f:
            next(f)  # Skip the header
            for line in f:
                out_degree, in_degree = map(int, line.strip().split())
                self.in_degrees.append(in_degree)
                self.out_degrees.append(out_degree)

    def calculate_metrics(self):
        self.avg_in_degree = np.mean(self.in_degrees)
        self.avg_out_degree = np.mean(self.out_degrees)
        self.var_in_degree = np.var(self.in_degrees)
        self.var_out_degree = np.var(self.out_degrees)

    def save_results(self):
        results = {
            "average_in_degree": self.avg_in_degree,
            "average_out_degree": self.avg_out_degree,
            "variance_in_degree": self.var_in_degree,
            "variance_out_degree": self.var_out_degree
        }
        with open(f"{self.filename}_results.json", 'w') as f:
            json.dump(results, f, indent=4)

    def plot_degree_distribution(self, xscale='linear', yscale='linear'):
        plt.figure(figsize=(12, 6))
        plt.hist(self.in_degrees, bins=range(min(self.in_degrees), max(self.in_degrees) + 1), alpha=0.5, label='In Degree')
        plt.hist(self.out_degrees, bins=range(min(self.out_degrees), max(self.out_degrees) + 1), alpha=0.5, label='Out Degree')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.legend(loc='upper right')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.savefig(f"{self.filename}_degree_distribution.png")
        plt.close()

    def plot_ccdf(self, xscale='log', yscale='log'):
        in_degree_counts = np.bincount(self.in_degrees)
        out_degree_counts = np.bincount(self.out_degrees)
        
        in_degree_sorted = np.arange(len(in_degree_counts))
        out_degree_sorted = np.arange(len(out_degree_counts))
        
        in_degree_ccdf = 1 - np.cumsum(in_degree_counts) / len(self.in_degrees)
        out_degree_ccdf = 1 - np.cumsum(out_degree_counts) / len(self.out_degrees)
        
        plt.figure(figsize=(12, 6))
        plt.plot(in_degree_sorted, in_degree_ccdf, marker='o', linestyle='none', label='In Degree')
        plt.plot(out_degree_sorted, out_degree_ccdf, marker='o', linestyle='none', label='Out Degree')
        plt.xlabel('Degree')
        plt.ylabel('CCDF')
        plt.title('Complementary Cumulative Degree Distribution (CCDF)')
        plt.legend(loc='upper right')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.savefig(f"{self.filename}_ccdf.png")
        plt.close()

    def plot_in_vs_out_degree(self, xscale='log', yscale='log'):
        plt.figure(figsize=(12, 6))
        plt.scatter(self.out_degrees, self.in_degrees, alpha=0.5)
        plt.xlabel('Out Degree')
        plt.ylabel('In Degree')
        plt.title('In Degree vs Out Degree')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.savefig(f"{self.filename}_in_vs_out_degree.png")
        plt.close()

# Example usage:
# analyzer = DegSeqAnalyzer('degree_sequence.txt')
if __name__ == '__main__':
    start = time.time()
    print("Starting analysis...")
    analyzer = DegSeqAnalyzer('deg_seq_test.txt')
    print("Done in", time.time() - start, "seconds")
