'''
This code we iterating over multiples search queries,
a query is a token we used for search by google search.
We do pagination on each query until results is present,
and extracting original size image + optionally saving locally'

Since we want to fine tune model with new images related to Vietnam,
such as: Vietnam people, Vietnam places,...
=> query tokens gonna be: "Vietnam",...
Queries is optional and can be changed base on users purposes.

'''

import urllib.request
from serpapi import GoogleSearch
import json
import os

def serpapi_get_google_images():
    image_results = []

    for query in ["Vietnam places", "Vietnam people", "Vietnam culture", "Vietnam history", "Vietnam traditions", "Vietnam food"]:
        # search query parameters
        params = {
            "engine": "google",               # search engine can change to Google, Bing, Yahoo, Naver, Baidu...
            "q": query,                       # search query
            "tbm": "isch",                    # image results
            "num": "100",                     # number of images per page
            "ijn": 0,                         # page number: 0 -> first page, 1 -> second...
            "api_key": "49924f99487362f105eb8eaa4f13d0c406be8044506e1d10f6edf9fb3d015bdb",                 # https://serpapi.com/manage-api-key
            # other query parameters: hl (lang), gl (country), etc
        }

        search = GoogleSearch(params)         # where data extraction happens

        images_is_present = True
        while images_is_present:
            results = search.get_dict()       # JSON -> Python dictionary

            # checks for "Google hasn't returned any results for this query."
            if "error" not in results:
                for image in results["images_results"]:
                    if image["original"] not in image_results:
                        image_results.append(image["original"])

                # update to the next page
                params["ijn"] += 1
            else:
                print(results["error"])
                images_is_present = False

    # -----------------------
    # Downloading images

    # make directory
    output_dir = "D:/SE2023-9.1/data/train/"
    os.makedirs(output_dir, exist_ok=True)

    for index, image_url in enumerate(image_results, start=1):
        print(f"Downloading {index} image...")

        opener=urllib.request.build_opener()
        opener.addheaders=[("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36")]
        urllib.request.install_opener(opener)

        file_name = f"{output_dir}original_size_img_{index}.jpg"
        urllib.request.urlretrieve(image_url, file_name)

    print(json.dumps(image_results, indent=2))
    print(len(image_results))

if __name__ == "__main__":
    # run function
    serpapi_get_google_images()