Sorti Commons photo stress test

`sorti_test_40s.mp4` is a deterministic 1280×720, 25 fps H.264 photo composite, with no burned-in labels. It is a synthetic motion/augmentation test using real photographic sources, rather than footage of physical objects moving in a real camera scene.

Run from this directory:

```bash
env -u PYTHONPATH ~/ps2_et/bin/python fetch_photos.py
env -u PYTHONPATH ~/ps2_et/bin/python make_video.py
env -u PYTHONPATH ~/ps2_et/bin/python review_video.py
env -u PYTHONPATH ~/ps2_et/bin/python eval_video.py \
  --pte ../mobilenet_v3_large_int8_xnnpack.pte \
  --labels ../labels.json --video sorti_test_40s.mp4 --truth truth.json
```

Dependencies: NumPy, Pillow, OpenCV, requests, torch and ExecuTorch in the Python environment; system ffmpeg with libx264. The evaluator writes `eval_<model stem>.json` beside its script. No model conversion is needed. Nine- and eleven-class metadata use the same `classes` and `input` schema as `../labels.json`; output length is validated against the labels.

There are 33 different Commons photos: three per waste class and three backgrounds. `selection.json` stores the reviewed source selection and normalized object crop rectangles. `credits.json` records author, licence, licence URL, Commons page, thumbnail URL, source SHA1 and download SHA256. `api_cache/` retains the Commons search/imageinfo responses. The fetcher requests width 640, downloads serially, retries rate limits with backoff, resizes locally to at most 640 px, and resumes without redownloading existing files. The API may supply a larger standard thumbnail URL even when the requested width is 640.

Sources come directly from Commons, not HF 4w4kt/waste-dataset-image, RealWaste, or TrashNet. The fetcher rejects Commons titles/pages already present in `../extra` credit manifests; no photos from `../extra/general_waste` are reused. Public dataset membership cannot be established from a Commons page alone; the provenance records make the selected sources auditable.

The video has eleven 3.8 s class segments plus two 2 s background segments, shuffled with seed 20261118. Twelve 0.4 s overlaps produce 41.0 s total. All three photos of each waste class appear in thirds of its segment. `timeline.json` records the source schedule and exact ROI. Each object has a reviewed crop, a GrabCut foreground mask trimmed to its visible bounds, and a feathered silhouette. `cutout_guides.json` supplies normalized silhouette constraints for seven difficult crops, including thin charger cables, attached packaging, pavement and can shadows. RGB is premultiplied by alpha before warping to avoid border fringes. Periodic position and scale changes, a full 360° rotation, mild perspective variation, exposure/contrast changes, occasional light motion blur, and sensor noise remain in place. Source photos, credit records, segment order and timings are unchanged.

Placement uses the geometric centre 60% square: `(424, 144, 432)` at 720p. The object's visible longest side must be 35–80% of 432 px (151.2–345.6 px), measured **after** masking, rotation, perspective and motion blur. Scale is normalized from the foreground bounds rather than the photographic rectangle; a one-pixel guard absorbs resampling rounding. The moving object's bbox centre stays inside both this square and the app ROI. Evaluation still imports the exact `centre_box(frame, 0.6)` from `../webcam_demo.py`, returning `(469, 114, 342)` to reserve the app's information panel. `timeline.json` retains that evaluation ROI.

`object_bboxes.json` logs bounds, centre, source index and longest-side size for every object-bearing segment frame. Each bbox uses alpha ≥0.5 after geometric transforms and blur; assertions enforce the size limits and centre containment. Crossfade frames have one record for each participating object, measured before crossfade opacity is applied. `video_review.jpg` shows all 30 object sources plus the three background segments; yellow outlines mark the 432 px placement square and green outlines mark the measured object bbox. These overlays and labels appear only in the review sheet.

`truth.json` contains half-open `[t0, t1)` intervals. Every crossfade is excluded entirely. The evaluator still infers every decoded frame, including fades, and saves top-1/confidence and segment membership. It imports the exact `preprocess` and `centre_box` implementations, converts BGR to RGB, and uses the supplied metadata's normalization and sizes.

For scoring, a new confirmation event occurs on the eighth consecutive prediction of the same class, with every confidence ≥0.70. Confidence below 0.70 or a different prediction breaks the streak. An uninterrupted long streak counts once; a newly established streak can confirm again. State resets at each truth interval; crossfade frames cannot contribute evidence. A segment succeeds when its first confirmation is correct. The report also includes all wrong confirmation events, time to first confirmation, time to first correct confirmation, and whether any correct confirmation occurred. A sustained `background` prediction is counted as background recognition for scoring; the live app suppresses disposal actions for that class.

Overall accuracy includes every labelled interval, including classes absent from a model's label file. Thus the v1 nine-class model necessarily misses `general_waste` and all `background` segments. Supported-class frame and segment accuracies are reported separately. Frame accuracy is weighted by scored frame count; segment accuracy weights every interval equally. This small composite stress test is not a representative estimate of deployment accuracy.

Verified run with `mobilenet_v3_large_int8_xnnpack.pte` and the supplied nine-class labels:

- 1,025 decoded frames; 905 labelled frames scored; 120 crossfade frames excluded.
- All 950 object-bearing segment frames passed size/centre assertions: longest side 152–345 px (35.19–79.86% of the 432 px square).
- Overall frame accuracy: 433/905 = 47.85%.
- Supported-class frame accuracy: 433/695 = 62.30%.
- First-confirmation segment accuracy: 7/13 = 53.85%; supported classes: 7/9 = 77.78%.
- 4 wrong confirmation events. No confirmation counts as segment failure.
- Correct first confirmations: light bulbs, clothes, e waste, plastic waste, metal waste, glass waste, paper waste.

The supplied nine-class model was exercised end to end. No eleven-class model was supplied for a second runtime check; the evaluator selects output names and preprocessing dynamically from either label schema.
