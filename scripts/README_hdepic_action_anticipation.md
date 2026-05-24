# HD-EPIC Action Anticipation Adapter

HD-EPIC can use the existing V-JEPA2 `action_anticipation_frozen` EK100 decoder without editing `vjepa2/`.
The adapter converts `HD_EPIC_Narrations.pkl` to EK100-style CSV files and optionally creates a lightweight
video link tree that matches the decoder's path convention.

## Convert Annotations

```powershell
python scripts\convert_hdepic_to_vjepa_csv.py `
  --annotations-pkl D:\path\to\hd-epic-annotations\narrations-and-action-segments\HD_EPIC_Narrations.pkl `
  --video-root D:\path\to\HD-EPIC `
  --output-dir D:\path\to\hdepic_vjepa_annotations `
  --link-root D:\path\to\hdepic_vjepa_videos `
  --link-method symlink `
  --val-participants P01
```

If symlinks are unavailable on Windows, use `--link-method hardlink` when the output tree is on the same
drive as the videos. Use `copy` only when storage is not a concern.

The generated `conversion_stats.json` records:

- `vjepa_base_path`: use this as `experiment.data.base_path`
- `vjepa_dataset`: keep this as `EK100`
- `vjepa_file_format`: keep this as `1`

## V-JEPA Config Fields

Use the generated train/val CSVs while keeping the original decoder settings:

```yaml
experiment:
  data:
    dataset: EK100
    file_format: 1
    base_path: D:\path\to\hdepic_vjepa_videos
    dataset_train: D:\path\to\hdepic_vjepa_annotations\HD_EPIC_train_vjepa.csv
    dataset_val: D:\path\to\hdepic_vjepa_annotations\HD_EPIC_val_vjepa.csv
```

The CSV `video_id` is rewritten from HD-EPIC's `P01-20240202-110250` format to `P01_20240202-110250`.
That lets the unmodified EK100 loader derive the participant folder with `video_id.split("_")[0]`.
