# How to use scraper

last update: 2022/11/04

```py
# use this code in if __name__ == '__main__':

from utils.scraper.pantip.pantip_scraper import TopicScraper, CommentScraper
from utils.scraper.pantip.save_utils import *
from utils.scraper.pantip.helper import *
from utils.scraper.pantip.dataframe_utils.dataset_preprocessing import *

# dataset setting
keyword = "sawano"
keyword_file = "sawano"

# file paths
# this pattern is created by scraper code
dataset_path = f"./data/{keyword_file}/"
topic_file = dataset_path + f"{keyword_file}.json"
comments_path = dataset_path + "/raw_comments/"

# initialise scraper
ts = TopicScraper()
ts.get_topic_detail = True
cs = CommentScraper()

# scrape data
print("Scrape data...save raw data to .json")
scrape_keyword(ts, cs, keyword, keyword_file, dataset_path)

# preprocess and save dataset file
print("Preprocess...save data to .csv")
raw_df = preprocess_keyword_dataset(topic_file, comments_path)
raw_df.to_csv(dataset_path + "raw_data.csv", index=False)
```
