# Training the ingredient detector

Fine-tunes a YOLO model to detect food ingredients in photos. The result is a
`best.pt` that the API loads via `YOLO_WEIGHTS`.

Nothing in this directory is imported by the app or shipped in the API image.

## Why this exists

The app ships two detectors behind the same port:

| Backend | Training | Vocabulary | Confidence | Speed |
|---|---|---|---|---|
| `llava` (default) | none | open — names anything it sees | fabricated (always 0.95) | Ollama round-trip |
| `yolo` | required | fixed to your label set | real per-box scores | ~50 ms on CPU |

Out of the box `yolo` loads a COCO checkpoint that knows **10 foods** (banana,
apple, orange, broccoli, carrot, sandwich, pizza, hot dog, donut, cake). That is
enough to prove the plumbing works and useless for real recipes. Everything
below is about replacing it.

## You need a dataset first

This is the real cost, and there is no way around it. YOLO is supervised: it
learns only classes you have drawn boxes for.

Budget roughly **150–300 labelled images per class** for usable accuracy. For
the 25 starter classes in `ingredients.yaml` that is 4,000–7,500 images.

Three routes, cheapest first:

1. **Adapt an existing dataset.** Roboflow Universe and Open Images V7 both
   carry food-detection sets with boxes already drawn. Export in *YOLOv8*
   format and remap the class list to match `ingredients.yaml`. Check the
   licence before using anything commercially.
2. **Label your own** with [Label Studio](https://labelstud.io/) or Roboflow.
   Slow, but the images match your users' actual photos, which matters more
   than raw dataset size.
3. **Auto-label, then correct.** Run the existing LLaVA detector over unlabelled
   photos to propose classes, then draw boxes by hand. Removes the "what is even
   in this picture" step; you still draw every box.

### Layout

```
datasets/ingredients/
  images/train/img001.jpg
  images/val/img900.jpg
  labels/train/img001.txt
  labels/val/img900.txt
```

Each `.txt` has one line per object, coordinates normalised to 0–1:

```
<class_index> <center_x> <center_y> <width> <height>
12 0.4812 0.5230 0.2140 0.3010
```

`class_index` refers to the `names:` map in `ingredients.yaml`. An image with
nothing to detect gets an empty `.txt` — do not omit the file.

Split roughly 80/20 train/val, and make sure the same dish photographed twice
does not land on both sides of the split. That leaks, and inflates your metrics.

## Train

Install the CUDA torch build into this environment only — the project pins CPU
wheels so the API image stays small:

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Then:

```bash
uv run python training/train_yolo.py --data training/ingredients.yaml
```

Useful flags: `--model yolo11s.pt` (bigger backbone), `--epochs 200`,
`--batch -1` (auto-size to your GPU), `--device cpu` (works, takes hours to days).

## Serve

```bash
export DETECTOR_BACKEND=yolo
export YOLO_WEIGHTS=runs/detect/ingredients/weights/best.pt
```

Class names are read from the checkpoint, so changing `ingredients.yaml` and
retraining needs no code change.

## Is it actually better?

Do not assume it is. LLaVA has an open vocabulary and needs no training; a
YOLO model trained on too little data will confidently detect the wrong
ingredient, which is worse than a vague answer in an app that feeds people with
allergies.

Before switching the default, hold out 100 photos neither model has seen and
compare both on the same set. Keep whichever wins.

## Where training should run

Not in GitHub Actions — hosted runners have no GPU. Options, in order of how
little setup they need: a Colab notebook against this repo, a rented cloud GPU
by the hour, or a self-hosted runner with a local card. Commit `best.pt` to a
GitHub Release rather than to git.
