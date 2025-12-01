import requests
import browser_cookie3

url = "https://sutdapac-my.sharepoint.com/personal/he_huang_mymail_sutd_edu_sg/_layouts/15/onedrive.aspx?viewid=d657f875%2Dd227%2D456c%2Daf24%2D3e48f620330c&ga=1&id=%2Fpersonal%2Fhe%5Fhuang%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FSUTD%2DTrafficQA%2DDataset%2Fraw%5Fvideos%2Fraw%5Fvideos%2Ezip&parent=%2Fpersonal%2Fhe%5Fhuang%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FSUTD%2DTrafficQA%2DDataset%2Fraw%5Fvideos"

cookies = browser_cookie3.chrome()   # load login cookies from Chrome

print("Downloading with authenticated cookies...")
resp = requests.get(url, cookies=cookies, stream=True)
resp.raise_for_status()

with open("raw_videos.zip", "wb") as f:
    for chunk in resp.iter_content(8192):
        f.write(chunk)

print("Done!")
