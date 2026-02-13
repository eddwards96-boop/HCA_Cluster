import base64
import os
from io import BytesIO

from PIL import Image


def _image_to_base64_png(image_path, max_size=80):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size), resample=Image.BILINEAR)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def generate_report_html(
    out_dir,
    dendrogram_path,
    image_paths,
    leaf_order,
    cluster_labels=None,
):
    os.makedirs(out_dir, exist_ok=True)

    dendro_b64 = _file_to_base64(dendrogram_path)
    rows = []
    for order_index, idx in enumerate(leaf_order, start=1):
        path = image_paths[idx]
        filename = os.path.basename(path)
        thumb_b64 = _image_to_base64_png(path, max_size=80)
        cluster_label = "-"
        if cluster_labels is not None:
            cluster_label = str(int(cluster_labels[idx]))
        rows.append((order_index, filename, cluster_label, thumb_b64))

    html_rows = []
    for order_index, filename, cluster_label, thumb_b64 in rows:
        html_rows.append(
            "<tr>"
            f"<td>{order_index}</td>"
            f"<td>{filename}</td>"
            f"<td>{cluster_label}</td>"
            f"<td><img src='data:image/png;base64,{thumb_b64}' /></td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ROI HCA Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    img {{ image-rendering: auto; }}
  </style>
</head>
<body>
  <h1>ROI HCA Report</h1>
  <h2>Dendrogram</h2>
  <img src="data:image/png;base64,{dendro_b64}" />
  <h2>Leaf Order</h2>
  <table>
    <tr>
      <th>Order</th>
      <th>Filename</th>
      <th>Cluster</th>
      <th>Thumbnail</th>
    </tr>
    {''.join(html_rows)}
  </table>
</body>
</html>
"""

    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path

