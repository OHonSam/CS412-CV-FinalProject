## 1. Download datasets
- Video dataset: SUTD Traffic Video QA dataset. Please download from: https://drive.google.com/drive/folders/1DFq5bJIfmL9w4meIUH7ZimNV4pMFCIzu?usp=drive_link with the destination folder as "VideoChat-Flash/datasets/videos"
- Question file: R2_test.jsonl (We have included this file in the repo under "VideoChat-Flash/datasets/R2_test.jsonl")
## 2. Download video encoder weights
Run the bash script in "VideoChat-Flash/misc/download_video_encoder.sh"
## 3. Install dependencies
```bash
pip install -r requirements.txt
```
## 4. Reproduce results
```bash
python main.py
```