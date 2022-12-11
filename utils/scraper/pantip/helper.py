from multiprocessing import Pool
from .save_utils import *
from .pantip_scraper import *
import pandas as pd
import numpy as np
import time


def multithread_scrape_comments(scraper,
                                topic_ids: list[str | int],
                                save_path: str,
                                pool_size: int = 4):
    """Use multithreading(multiprocessing) to speed up scraping speed
    sprape for comments of multiple topics simultaneously

    Args:
        scraper (_type_): comment scrapers
        topic_ids (list[str  |  int]): a list of topic ids
        save_path (str): path to save comments json
        pool_size (int, optional): CPU's core to use. Defaults to 4.
    """

    # each element is a list of responses for a topic
    with Pool(pool_size) as p:
        responses = p.map(scraper.scrape, topic_ids)

    folder_ = [save_path] * len(responses)

    with Pool(pool_size) as p:
        p.starmap(save_comments_json, zip(responses, folder_))


def scrape_keyword(topic_scraper, comment_scraper, keyword: str, keyword_file: str,
                   dataset_path: str):

    topics_file = dataset_path + f"{keyword_file}.json"

    ts = topic_scraper
    cs = comment_scraper

    topic_responses = ts.scrape(keyword)
    time.sleep(np.random.randint(1, 3))
    save_topics_json(topic_responses, keyword_file, dataset_path)
    print(f"Topics file is saved at {topics_file}")

    # load topic file to process
    with open(topics_file) as f:
        topics_json = json.load(f)

    topics_df = pd.DataFrame(topics_json)
    ids = topics_df['id'].drop_duplicates()

    num_topics = len(ids)
    print(f"Found {num_topics} unique topics")

    # batch + multithread scraping
    # for large size query
    batch_size = 32
    n_batches = int(np.ceil(num_topics / batch_size))
    for i in range(0, n_batches):
        if (end := (i + 1) * batch_size) > num_topics:
            end = num_topics
            batch_ids = ids[i * batch_size:]
        else:
            batch_ids = ids[i * batch_size:end]

        print(f"{i * batch_size}..{end}")
        multithread_scrape_comments(cs, batch_ids, dataset_path)
        time.sleep(np.random.randint(1, 5))