# run this codeblock to download everything from bensound
# you'll need to mount your google drive to store the data
# it'll save the mp3s in a folder that's in your My Drive folder called mp3_instrumental
import requests
from lxml import html
import re
import os
# from google.colab import drive
# drive.mount('/content/gdrive')


if __name__ == '__main__':
    if not os.path.exists('gdrive/My\ Drive/mp3_instrumental'):
        os.makedirs('gdrive/My\ Drive/mp3_instrumental')

    royalty_free_string = 'https://www.bensound.com/royalty-free-music/'
    url_base = 'https://www.bensound.com'
    pat = re.compile(r'/bensound-music/.*mp3')
    counter = 1

    for j in range(1, 27):
        req_url = royalty_free_string + str(j)
        page = requests.get(req_url)
        routes = pat.findall(page.text)  

        for route in routes:
            music_link = url_base + route
            os.system('curl {} -o gdrive/My\ Drive/mp3_instrumental/track_{}.mp3'.format(music_link, counter))
            print(music_link)
            counter += 1
            print('%i songs processed' % counter)

        x = pat.findall(page.text)
        print(x)
