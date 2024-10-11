import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import rplanpy


def test_functions(file: str, out_file: str = 'example_graph.png', plot_original: bool = True) -> None:

    data = rplanpy.data.RplanData(file)
    data.set_graph()
    G = data.get_graph()
    print(G.graph)
    print(G.nodes.data())
    print(G.edges.data())
    
    # for node in G.nodes:
    #     print(f"category: {G.nodes[node]['category']}")

    room_categories = [G.nodes[node]['category'] for node in G.nodes]
    print(room_categories)

    for u,v,edg_data in G.edges(data=True):
        src = u
        target = v
        location = edg_data['location']
        door = edg_data['door']
        print(f"src:target \t {src}  :  {target}, location={location} , door={door}")

    plt.imshow(data.image,origin='lower')
    ax = plt.gca()

    for node in G.nodes:
        min_row,min_col,max_row,max_col = G.nodes[node]['bounding_box'];

        print(f"room bounding box: {min_row} {min_col} {max_row} {max_col}")
        # rect = Rectangle((min_col, min_row), max_col-min_col, max_row-min_row,linewidth=1, edgecolor='b')
        # ax.add_patch(rect)


    min_row,min_col,max_row,max_col = G.graph['site_bounding_box']
    print(f"site bounding box : {min_row} {min_col} {max_row} {max_col}")

    rect = Rectangle((min_col ,min_row),max_col-min_col , max_row-min_row,linewidth=2,edgecolor='r',facecolor='none')

    ax.add_patch(rect)
    plt.show()

    




    ncols = 3 if plot_original else 2
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))
    if plot_original:
        image = imageio.imread(file)
        ax[0].imshow(image,origin='lower')
        ax[0].axis("on")
        ax[0].set_title("Original image")
    rplanpy.plot.plot_floorplan(data, ax=ax[plot_original+0], title="Rooms and doors")
    ax = rplanpy.plot.plot_floorplan_graph(
        data=data, with_colors=True, edge_label='door', ax=ax[plot_original+1],
        title="Building graph"
    )
    plt.tight_layout()
    plt.savefig(out_file)

    plt.show()


if __name__ == '__main__':
    file = 'example.png'
    file2 = 'dataset/floorplan_dataset/0.png'
    test_functions(file2, out_file='file2_graph.png', plot_original=True)
