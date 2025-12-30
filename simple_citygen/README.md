Simple simplified roadmap generator using the "organic" growth rule inspired by procedural_city_generation.

Usage:

1. (Optional) create a virtualenv and install requirements:

```bash
conda create -n simple_citygen python=3.10
conda activate simple_citygen
pip install -r requirements.txt
```

2. Run the demo:

```bash
python main.py
```

This will produce `simple_roadmap.png` in the current directory.

Scene generation from data only:

```bash
python scripts/generate_scene.py --data-dir ./data --materials --out ./data/scene/placed_scene.obj
```

Preview without exporting (low memory):

```bash
python scripts/generate_scene.py --data-dir ./data --visualize --no-materials
```
