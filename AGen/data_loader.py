import numpy as np
from PIL import Image
from lxml import html
import requests
from bs4 import BeautifulSoup
import json
import os


# 4cd635ad71c0140175152a95c0e3974f9cc4bbdc

class DataLib:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.unicode_id_link = "https://emoji-api.com/emojis?access_key=4cd635ad71c0140175152a95c0e3974f9cc4bbdc"
        self.unicode_ids = self.load_unicode_lookup()

    def load_unicode_lookup(self):
        unicode_ids = {}
        result_html = BeautifulSoup(requests.get(self.unicode_id_link).text, "html.parser")
        for emoji in json.loads(str(result_html)):
            if len(emoji['codePoint'].split()) <= 2:
                unicode_ids[emoji['codePoint']] = emoji['unicodeName']

        return unicode_ids

print(
    DataLib("?").unicode_ids["1F1E7 1F1EE"]
)