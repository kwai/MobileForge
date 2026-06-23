"""
Page Graph Visualizer - Reuse existing code to generate interactive page graph visualization
"""
import json
import os
from typing import Dict, Any
import base64
from pathlib import Path

class PageGraphVisualizer:
    """Page graph visualizer"""

    def __init__(self):
        self.template_dir = os.path.join(os.path.dirname(__file__), "visualization_templates")
        self.ensure_template_dir()

    def ensure_template_dir(self):
        """Ensure template directory exists"""
        os.makedirs(self.template_dir, exist_ok=True)

    def visualize_graph(self, graph_data: Dict[str, Any], output_file: str = None) -> str:
        """Generate interactive page graph visualization"""
        if output_file is None:
            app_package = graph_data.get("app_package", "unknown")
            output_file = f"page_graph_{app_package}.html"

        # Prepare visualization data
        vis_data = self._prepare_visualization_data(graph_data)

        # Generate HTML visualization
        html_content = self._generate_html_visualization(vis_data, graph_data)

        # Save file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Page graph visualization generated: {output_file}")
        return os.path.abspath(output_file)

    def _prepare_visualization_data(self, graph_data: Dict[str, Any]) -> Dict:
        """Prepare visualization data format"""
        pages = graph_data.get("pages", {})
        transitions = graph_data.get("transitions", {})

        # Convert to D3.js format
        nodes = []
        links = []

        # Process page nodes
        for page_id, page_data in pages.items():
            # Process screenshots
            screenshot_data = self._encode_screenshot(page_data.get("representative_screenshot"))

            node = {
                "id": page_id,
                "page_type": page_data.get("page_type", "custom_page"),
                "activity_name": page_data.get("activity_name", "unknown"),
                "visit_count": page_data.get("visit_count", 0),
                "confidence": page_data.get("confidence", 0.5),
                "ui_element_count": page_data.get("ui_element_count", 0),
                "key_features": page_data.get("key_features", []),
                "screenshot": screenshot_data,
                "common_actions": page_data.get("common_actions", [])
            }
            nodes.append(node)

        # Process transition relationships
        for trans_id, trans_data in transitions.items():
            link = {
                "source": trans_data.get("from_page_id"),
                "target": trans_data.get("to_page_id"),
                "action_type": trans_data.get("action_type", "unknown"),
                "action_target": trans_data.get("action_target", "unknown"),
                "success_rate": trans_data.get("success_count", 0) / max(trans_data.get("total_count", 1), 1),
                "total_count": trans_data.get("total_count", 0),
                "id": trans_id
            }
            links.append(link)

        return {
            "nodes": nodes,
            "links": links,
            "statistics": graph_data.get("statistics", {})
        }

    def _encode_screenshot(self, screenshot_path: str) -> str:
        """Encode screenshot as base64 for web display"""
        if not screenshot_path or not os.path.exists(screenshot_path):
            return ""

        try:
            with open(screenshot_path, "rb") as img_file:
                img_data = img_file.read()
                return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            print(f"Failed to encode screenshot: {e}")
            return ""

    def _generate_html_visualization(self, vis_data: Dict, graph_data: Dict) -> str:
        """Generate HTML visualization page"""
        app_package = graph_data.get("app_package", "unknown")
        statistics = vis_data.get("statistics", {})

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Graph Visualization - {app_package}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            padding: 20px;
            border-bottom: 1px solid #ddd;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px 8px 0 0;
        }}
        .controls {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
            background: #fafafa;
        }}
        .graph-container {{
            position: relative;
            height: 600px;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            width: 300px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            display: none;
        }}
        .statistics {{
            padding: 20px;
            border-top: 1px solid #eee;
            background: #fafafa;
        }}
        .stat-item {{
            display: inline-block;
            margin: 0 20px 10px 0;
            padding: 10px 15px;
            background: white;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            cursor: pointer;
        }}
        .node-label {{
            font-size: 10px;
            pointer-events: none;
            text-anchor: middle;
            fill: #333;
        }}
        .screenshot {{
            max-width: 200px;
            max-height: 300px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        button {{
            padding: 8px 16px;
            margin: 0 5px;
            border: none;
            border-radius: 4px;
            background: #667eea;
            color: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #5a6fd8;
        }}
        select {{
            padding: 8px;
            margin: 0 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Page Graph Visualization - {app_package}</h1>
            <p>Explore app page structure and navigation relationships</p>
        </div>

        <div class="controls">
            <button onclick="resetZoom()">Reset View</button>
            <button onclick="toggleLabels()">Toggle Labels</button>
            <select id="layoutSelect" onchange="changeLayout()">
                <option value="force">Force Layout</option>
                <option value="radial">Radial Layout</option>
                <option value="hierarchical">Hierarchical Layout</option>
            </select>
            <select id="filterSelect" onchange="filterNodes()">
                <option value="all">Show All</option>
                <option value="main_page">Main Page</option>
                <option value="settings_page">Settings Page</option>
                <option value="list_page">List Page</option>
                <option value="detail_page">Detail Page</option>
            </select>
        </div>

        <div class="graph-container">
            <svg id="graph"></svg>
            <div id="infoPanel" class="info-panel">
                <h3>Page Details</h3>
                <div id="panelContent"></div>
            </div>
        </div>

        <div class="statistics">
            <h3>Statistics</h3>
            <div class="stat-item">
                <strong>Total Pages:</strong> {statistics.get('total_pages', 0)}
            </div>
            <div class="stat-item">
                <strong>Total Transitions:</strong> {statistics.get('total_transitions', 0)}
            </div>
            <div class="stat-item">
                <strong>Processed Trajectories:</strong> {statistics.get('processed_trajectories', 0)}
            </div>
            <div class="stat-item">
                <strong>Average Success Rate:</strong> {statistics.get('avg_success_rate', 0):.2%}
            </div>
        </div>
    </div>

    <script>
        // Visualization data
        const graphData = {json.dumps(vis_data, ensure_ascii=False, indent=2)};

        // Page type color mapping
        const pageTypeColors = {{
            'main_page': '#4CAF50',
            'settings_page': '#FF9800',
            'profile_page': '#2196F3',
            'login_page': '#F44336',
            'search_page': '#9C27B0',
            'list_page': '#607D8B',
            'detail_page': '#795548',
            'edit_page': '#E91E63',
            'help_page': '#009688',
            'about_page': '#3F51B5',
            'custom_page': '#9E9E9E'
        }};

        // SVG settings
        const svg = d3.select("#graph");
        const container = d3.select(".graph-container");
        const width = container.node().getBoundingClientRect().width;
        const height = container.node().getBoundingClientRect().height;

        svg.attr("width", width).attr("height", height);

        const g = svg.append("g");

        // Zoom functionality
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Force simulation
        let simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        // Draw links
        const link = g.append("g")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link")
            .style("stroke-width", d => Math.max(1, d.success_rate * 5))
            .style("stroke-opacity", d => 0.3 + d.success_rate * 0.7)
            .on("mouseover", showLinkTooltip)
            .on("mouseout", hideTooltip);

        // Draw nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.max(10, Math.log(d.visit_count + 1) * 5))
            .style("fill", d => pageTypeColors[d.page_type] || '#999')
            .style("opacity", d => 0.7 + d.confidence * 0.3)
            .on("click", showNodeDetails)
            .on("mouseover", showNodeTooltip)
            .on("mouseout", hideTooltip)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Node labels
        let showLabels = true;
        const labels = g.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.page_type.replace('_page', ''))
            .attr("dy", -15);

        // Simulation update
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});

        // Interactive functionality
        function showNodeDetails(event, d) {{
            const panel = document.getElementById('infoPanel');
            const content = document.getElementById('panelContent');

            let screenshotHtml = '';
            if (d.screenshot) {{
                screenshotHtml = `<img src="data:image/png;base64,${{d.screenshot}}" class="screenshot" alt="Page Screenshot">`;
            }}

            content.innerHTML = `
                <h4>${{d.page_type.replace('_', ' ')}}</h4>
                <p><strong>Activity:</strong> ${{d.activity_name}}</p>
                <p><strong>Visit Count:</strong> ${{d.visit_count}}</p>
                <p><strong>Confidence:</strong> ${{(d.confidence * 100).toFixed(1)}}%</p>
                <p><strong>UI Elements:</strong> ${{d.ui_element_count}}</p>
                <p><strong>Key Features:</strong> ${{d.key_features.join(', ')}}</p>
                <p><strong>Common Actions:</strong> ${{d.common_actions.join(', ')}}</p>
                ${{screenshotHtml}}
            `;

            panel.style.display = 'block';
        }}

        function showNodeTooltip(event, d) {{
            // Simple tooltip implementation
            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("position", "absolute")
                .style("background", "rgba(0,0,0,0.8)")
                .style("color", "white")
                .style("padding", "8px")
                .style("border-radius", "4px")
                .style("pointer-events", "none")
                .style("opacity", 0);

            tooltip.transition().duration(200).style("opacity", 1);
            tooltip.html(`${{d.page_type}}<br/>Visits: ${{d.visit_count}} times`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }}

        function showLinkTooltip(event, d) {{
            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("positionsolute")
                .style("background", "rgba(0,0,0,0.8)")
                .style("color", "white")
                .style("padding", "8px")
                .style("border-radius", "4px")
                .style("pointer-events", "none")
                .style("opacity", 0);

            tooltip.transition().duration(200).style("opacity", 1);
            tooltip.html(`${{d.action_type}}<br/>Success Rate: ${{(d.success_rate * 100).toFixed(1)}}%`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }}

        function hideTooltip() {{
            d3.selectAll(".tooltip").remove();
        }}

        function resetZoom() {{
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
        }}

        function toggleLabels() {{
            showLabels = !showLabels;
            labels.style("display", showLabels ? "block" : "none");
        }}

        function changeLayout() {{
            const layout = document.getElementById('layoutSelect').value;
            // Different layout algorithms can be implemented here
            if (layout === 'radial') {{
                // Radial layout logic
                console.log('Switch to radial layout');
            }} else if (layout === 'hierarchical') {{
                // Hierarchical layout logic
                console.log('Switch to hierarchical layout');
            }}
        }}

        function filterNodes() {{
            const filter = document.getElementById('filterSelect').value;

            node.style("opacity", d => {{
                if (filter === 'all' || d.page_type === filter) {{
                    return 0.7 + d.confidence * 0.3;
                }} else {{
                    return 0.1;
                }}
            }});

            labels.style("opacity", d => {{
                if (filter === 'all' || d.page_type === filter) {{
                    return 1;
                }} else {{
                    return 0.1;
                }}
            }});
        }}

        // Drag functionality
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Click empty area to hide info panel
        svg.on("click", function(event) {{
            if (event.target === this) {{
                document.getElementById('infoPanel').style.display = 'none';
            }}
        }});
    </script>
</body>
</html>
        """

        return html_template