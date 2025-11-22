#!/usr/bin/env python3
"""
visualize_cluster_means.py
--------------------------
Generate HTML visualization of cluster means table.
"""

import pandas as pd
import sys


def generate_html(csv_path: str = "cluster_means.csv") -> str:
    """Generate HTML visualization of cluster means table."""

    # Read CSV
    df = pd.read_csv(csv_path, index_col=0)

    # Get max value for color scaling
    max_val = df.values.max()
    min_val = df.values.min()

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cluster Means Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        .function-header {{
            background-color: #e9ecef;
            font-weight: bold;
            text-align: left;
            padding-left: 15px;
        }}
        .value-cell {{
            font-weight: bold;
            color: white;
            position: relative;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .legend-scale {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .color-bar {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #fff3cd, #ffeaa7, #fdcb6e, #e17055, #d63031);
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Cluster Means by Function</h1>
        <table>
            <thead>
                <tr>
                    <th>Function</th>"""

    # Add cluster column headers
    for col in df.columns:
        html += f"<th>{col}</th>"

    html += """
                </tr>
            </thead>
            <tbody>"""

    # Add data rows
    for function, row in df.iterrows():
        html += f"""
                <tr>
                    <td class="function-header">{function}</td>"""

        for value in row:
            # Calculate color intensity based on value
            if max_val > min_val:
                intensity = (value - min_val) / (max_val - min_val)
            else:
                intensity = 0.5

            # Color scale from light yellow to dark red
            if intensity < 0.2:
                color = f"rgb(255, 243, 205)"  # Light yellow
                text_color = "#333"
            elif intensity < 0.4:
                color = f"rgb(255, 234, 167)"  # Yellow
                text_color = "#333"
            elif intensity < 0.6:
                color = f"rgb(253, 203, 110)"  # Orange
                text_color = "#333"
            elif intensity < 0.8:
                color = f"rgb(225, 112, 85)"  # Red-orange
                text_color = "white"
            else:
                color = f"rgb(214, 48, 49)"  # Dark red
                text_color = "white"

            html += f"""<td class="value-cell" style="background-color: {color}; color: {text_color};">{value:.3f}</td>"""

        html += "</tr>"

    html += f"""
            </tbody>
        </table>
        
        <div class="legend">
            <div class="legend-title">Color Scale</div>
            <div class="legend-scale">
                <span>Low ({min_val:.3f})</span>
                <div class="color-bar"></div>
                <span>High ({max_val:.3f})</span>
            </div>
            <p><strong>Interpretation:</strong> Higher values (darker colors) indicate that codes of this function appear more frequently in articles assigned to this cluster.</p>
        </div>
        
        <div style="margin-top: 20px; padding: 10px; background-color: #e3f2fd; border-radius: 5px;">
            <strong>Data source:</strong> {csv_path}<br>
            <strong>Functions:</strong> {len(df)} frame functions<br>
            <strong>Clusters:</strong> {len(df.columns)} clusters identified
        </div>
    </div>
</body>
</html>"""

    return html


def save_html(
    csv_path: str = "cluster_means.csv", output_path: str = "cluster_means.html"
):
    """Generate and save HTML visualization."""
    try:
        html_content = generate_html(csv_path)

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"HTML visualization saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find CSV file: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main function."""
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "cluster_means.csv"
    output_path = csv_path.replace(".csv", ".html")

    save_html(csv_path, output_path)


if __name__ == "__main__":
    main()
