from py_script.dataset import GMD
import torch_geometric.utils as pygutils

if __name__ == "__main__":
    dataset = GMD("./test/data", name="epri21")
    print(dataset)
    data = dataset[0]
    net = pygutils.to_networkx(data)
    print(f"# nodes {len(net.nodes)}",
          f"# of edges {len(net.edges)}")