import zipfile
import os
from loguru import logger

source = r"D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/videos_obj_tracking/videos_obj_tracking"
zip_path = r"C:/Users/Admin/videos_obj_tracking.zip"  # external drive!

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(source):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, arcname=os.path.relpath(full, source))
            logger.info(f"Zipped: {full}")
            
# zip -r "C:/videos_obj_tracking.zip" "D:/APCS/2025-2026/Computer_Vision/PROJECT/CS412-CV-FinalProject/sutd-traffic-video-qa/videos_obj_tracking/videos_obj_tracking"