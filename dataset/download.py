import os
import requests
import urllib.parse

API_KEY = "YOUR_BING_KEY"
ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

def download_images(query, folder, count=50):
    os.makedirs(folder, exist_ok=True)
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": query, "count": count}

    response = requests.get(ENDPOINT, headers=headers, params=params)

    # If Bing returns an error, print it and stop
    if "value" not in response.json():
        print("\n❌ Bing API Error for query:", query)
        print(response.json())
        return

    data = response.json()

    for i, img in enumerate(data["value"]):
        try:
            url = img["contentUrl"]
            ext = ".jpg"
            path = os.path.join(folder, f"{i}{ext}")
            img_data = requests.get(url, timeout=5).content
            with open(path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            print("Skipping image:", e)

if __name__ == "__main__":
    categories = ["fade haircut", "buzz cut", "mullet haircut", "bob haircut"]
    for cat in categories:
        folder = f"images/{cat.replace(' ', '_')}"
        download_images(cat, folder)
