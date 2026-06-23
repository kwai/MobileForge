# UI Transition Graph Builder

This tool builds a UI Transition Graph (UTG) from MobileForge exploration trajectories.

## Install

Install the dependencies required by the exploration stack, then install the graph utilities:

```bash
pip install zstandard networkx
```

## Usage

```bash
python graph_analysis/graph_builder.py \
  -trajectory_dir <path_to_exploration_output> \
  -package_name <app_package_name> \
  -output_dir <output_directory_for_utg>
```

Arguments:

- `-trajectory_dir`: directory containing `.pkl.zst` trajectory files, for example `exploration_output/net.osmand`.
- `-package_name`: app package name, for example `net.osmand`.
- `-output_dir`: output directory for `utg.js`; defaults to `./utg_results`.

Example:

```bash
python graph_analysis/graph_builder.py \
  -trajectory_dir exploration_output/net.osmand \
  -package_name net.osmand \
  -output_dir utg_visualization
```

## Output

The script writes `utg.js`, which contains the graph nodes and edges needed for visualization. Place the generated `utg.js` next to a simple HTML page that loads `vis-network`.

Minimal `index.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <title>UI Transition Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    #mynetwork {
      width: 100%;
      height: 800px;
      border: 1px solid lightgray;
    }
  </style>
</head>
<body>
  <div id="mynetwork"></div>
  <script src="./utg.js"></script>
  <script>
    const nodes = new vis.DataSet(utg.nodes);
    const edges = new vis.DataSet(utg.edges);
    const network = new vis.Network(
      document.getElementById("mynetwork"),
      { nodes, edges },
      {
        nodes: { shape: "dot", size: 16, font: { size: 12 } },
        edges: { width: 2, arrows: "to", smooth: { type: "continuous" } },
        physics: { enabled: true }
      }
    );
  </script>
</body>
</html>
```
