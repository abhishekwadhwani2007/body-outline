# Body Outline

A computer vision module that draws a **neon glowing body outline** around a person in any photo.

Built from scratch as the overlay engine for **CameraCoach** — an app that lets you pick a reference photo and align your own pose to it in real time.

![Demo](testImage_overlay.png)

---

## Why This Exists

CameraCoach needed one core feature: overlay a glowing silhouette of a reference pose over a live camera feed so the user can match their body position to it.

The idea sounds simple. Finding a way to do it cleanly was not.

There was no existing library or open-source project that extracted a smooth, accurate body outline from an arbitrary photo. Segmentation APIs give you a mask, not an outline. Edge detectors give you noise. Skeleton overlays skip the full body shape. None of them produced what we needed — a clean, smooth, glow-edged silhouette that works on real-world photos with complex backgrounds.

So we built this from the ground up.

---

## How It Works

The pipeline runs four steps:

```
Input photo
    ↓
MediaPipe TFLite pose model
    → 39 body landmarks + coarse segmentation mask
    ↓
Skeleton mask (from landmarks) merged with segmentation mask
    ↓
GrabCut refinement (sharpens edges against real image pixels)
    ↓
Smoothed contour extraction + multi-layer neon glow
    ↓
Output photo with outline
```

### Why landmarks + segmentation together?

The segmentation mask from the model is coarse — it gives a rough blob. On its own, it misses thin limbs and loses detail at the edges. The landmark-based skeleton fills in the gaps. Together they're much more reliable than either alone.

### What GrabCut does here

GrabCut uses the merged mask as a starting hint, then looks at the actual pixel colors in the image to find the real foreground edge. It's the step that makes the outline follow the actual body shape rather than just the model's approximation.

---

## What Was Broken and How We Fixed It

Building this took a lot of iteration. Here's what actually went wrong and what solved each problem.

### Arms getting cut out of the outline

**Problem:** GrabCut classified thin arm regions as background and erased them. Outstretched arms were especially bad — they'd just disappear from the outline.

**Fix:** Before running GrabCut, we paint thick strokes along both arm chains (shoulder → elbow → wrist) into a "force foreground" mask. GrabCut respects this and keeps those pixels as foreground no matter what.

### Held objects creating a bump or hump

**Problem:** When someone holds something in front of their body (like a ball or prop), the mask would include it in a messy, distorted way — creating a weird lump in the outline where it didn't belong.

**Fix:** We added a held-object detector that looks for dark, saturated blobs near the wrists. When it finds one, it fits a clean ellipse over it and adds it properly to the mask. The result is a smooth, intentional shape instead of a random protrusion.

### Hair making the head outline jagged

**Problem:** Loose hair, ponytails, and flyaways create very noisy mask edges at the top of the head. The outline would look spiky and irregular there.

**Fix:** We apply morphological smoothing only to the head region (bounded by the ear and shoulder landmarks). We use a connected component filter seeded from the head position so we only keep regions that actually belong to the head — not random noise nearby.

### Face disappearing under the glow

**Problem:** The glow outline is drawn along the full silhouette edge, which includes the face. That meant the person's face got covered by the green glow.

**Fix:** After drawing the outline, we use a distance transform inside the head region to create a soft alpha blend that gradually restores the original face pixels. The glow stays on the edges — the face stays visible.

### Edges looking like a polygon

**Problem:** Raw OpenCV contours follow pixel grid boundaries exactly. At normal viewing size this looks like a jagged polygon rather than a body shape.

**Fix:** Each contour is resampled to evenly-spaced points along its perimeter, then smoothed with a wrapped 1D Gaussian filter. This produces organic, curved edges that actually look like a person.

---

## Output Quality

Results vary based on the input photo.

Photos close to the model's training distribution (person standing, clear full-body view, decent lighting) → clean, accurate outline.

Unusual poses, heavy occlusion, motion blur, or cluttered backgrounds → inconsistent results.

This is a known limitation. The underlying model (`pose_landmark_full.tflite`) was trained by Google on a specific distribution of images. When your photo matches that distribution, it works well. When it doesn't, the segmentation mask starts out noisy and the pipeline can only partially compensate.

**What we're planning to improve:**
- Custom segmentation training on photos closer to CameraCoach's real use cases
- Confidence-based fallback to a simpler mask when pose score is low
- Exploring SAM (Segment Anything) as an alternative segmentation backend

---

## Setup

Python 3.10+

```bash
pip install -r requirements.txt
```

**Model file:** Download `pose_landmark_full.tflite` from [Google MediaPipe Models](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task) and place it in the project root. It's not included here because of its size (6.1 MB).

---

## Usage

Copy `.env` and set your image path:

```bash
cp .env .env  # it's already there, just edit it
```

```env
INPUT_IMAGE=your_photo.jpg
OUTPUT_IMAGE=your_photo_overlay.png
```

Then run:

```bash
python pose_outline.py
```

Or pass paths directly:

```bash
python pose_outline.py your_photo.jpg output.png
```

**Python API:**

```python
from pose_outline import process

process("input.jpg", "output.png")
```

---

## Project Structure

```
body-outline/
├── pose_outline.py           # The whole pipeline — one file
├── pose_landmark_full.tflite # MediaPipe pose model (not in git — download separately)
├── requirements.txt
├── .env                      # Your image paths — not committed
├── .env.example              # Template for .env
├── testImage.png             # Sample input
└── testImage_overlay.png     # Sample output
```

---

## Part of CameraCoach

CameraCoach is an app for getting the exact pose you want in a photo. You pick a reference image, and the app overlays a glowing silhouette of that pose over your live camera view so you can align your body to match it.

This module is the piece that generates that silhouette.

---

## License

MIT
