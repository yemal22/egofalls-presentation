# EgoFalls video demo

Upload a short video and run the saved Model 1 motion classifier. The interface uses the original notebook04 feature extraction: 4 fps scan, 160 px width, up to 16 uniform adjacent frame pairs, RAFT-small with six updates, eight motion summaries, saved scaler/classifier and fold-specific threshold.

## Run on Kaggle

1. Import `EgoFalls_07_Video_Demo.ipynb` into a new Kaggle notebook.
2. Enable GPU and Internet.
3. Attach outputs from `fanelp/egofalls-06-video-lopo-comparison` (seven `test_*_motion.joblib` files) and `yemal1n/video-only-training` (`feature_config.json`).
4. Run all cells. Open the temporary Gradio URL printed by the last cell. Keep the session running.
5. Choose the checkpoint whose held-out test person matches your demo clip. Upload the video and analyze it.

No retraining is performed. Official RAFT and ResNet weights download on first launch; the original full extractor is preserved for compatibility. FFmpeg and ffprobe must be installed (available in Kaggle). A local GPU runtime can use `EGOFALLS_INPUT_ROOT` to point at trusted downloaded outputs and run `python app.py` after installing requirements.

## Evidence and limits

This is whole-clip inference, not a live-camera detector or fall-time localizer. Model scores are not calibrated clinical risks. Test with existing staged recordings. Report held-out successes and mistakes. The published benchmark has 86.9% pooled recall, 77.7% mean-participant recall, 39.1% worst-person recall and 43.4 false alarms per 100 non-fall clips.

Training configuration fingerprints and official weight hashes are checked. Differences in Torch/TorchVision versions are disclosed in the result. Checkpoint deserialization is limited to attached project outputs; only attach files you trust.

Validation completed here: Python syntax and notebook structure. GPU end-to-end inference and the public Gradio link require a running Kaggle GPU session and have not yet been verified.
