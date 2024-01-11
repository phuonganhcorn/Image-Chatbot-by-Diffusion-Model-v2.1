import os
import urllib.request
import json
from bs4 import BeautifulSoup

def get_soup(url, header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=header)).read(), 'html.parser')

def scrape_images(query_list):
    for query in query_list:
        image_type = "Action"
        query = query.split()
        query = '+'.join(query)
        url = "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"
        print(url)
        
        # Specify the directory to save the images
        DIR = "D:/SE2023-9.1/data/train/"
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        
        header = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
        }
        soup = get_soup(url, header)

        actual_images = []  # contains the link for large original images and the type of image
        for a in soup.find_all("div", {"class": "rg_meta"}):
            link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            actual_images.append((link, Type))

        print("There are a total of", len(actual_images), "images")

        query_dir = os.path.join(DIR, query.split()[0])
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)

        # Download and save the images
        for i, (img, Type) in enumerate(actual_images):
            try:
                req = urllib.request.Request(img, headers={'User-Agent': header})
                raw_img = urllib.request.urlopen(req).read()

                cntr = len([i for i in os.listdir(query_dir) if image_type in i]) + 1
                print(cntr)
                if len(Type) == 0:
                    f = open(os.path.join(query_dir, image_type + "_" + str(cntr) + ".jpg"), 'wb')
                else:
                    f = open(os.path.join(query_dir, image_type + "_" + str(cntr) + "." + Type), 'wb')

                f.write(raw_img)
                f.close()
            except Exception as e:
                print("Could not load: " + img)
                print(e)

if __name__ == "__main__":
    query_list = ["Vietnam places", "Vietnam people", "Vietnam culture", "Vietnam history", "Vietnam traditions", "Vietnam food"]
    scrape_images(query_list)