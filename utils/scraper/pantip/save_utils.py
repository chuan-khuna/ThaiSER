import json
import os


def save_comments_json(scrape_result: list[dict], output_folder: str = "."):
    """select only comment data and save comments json into `/output_path/raw_comments/<topic_id>`

    Args:
        scrape_result (list[dict]): raw result from API scraping (list of pagination responses)
        output_folder (str, optional): output folder for the dataset. Defaults to ".".
    """

    comments = []

    # check if a topic have comments
    # todo: refactor, why the api doesn't return comments in that topic
    if len(scrape_result) > 0:
        try:
            topic_id = scrape_result[0]['paging']['topic_id']
        except Exception:
            return
    else:
        return

    for batch in scrape_result:
        if 'comments' in batch.keys():
            comments += batch['comments']

    output_folder = output_folder + "/raw_comments/"
    if not os.path.exists(f"{output_folder}"):
        os.makedirs(f"{output_folder}", exist_ok=True)

    output_file = f"{output_folder}/{topic_id}.json"

    with open(output_file, "w") as f:
        f.write(json.dumps(comments, ensure_ascii=False, indent=2))
        # print(output_file)


def save_topics_json(scrape_result: list[dict], keyword: str, output_folder: str = "."):
    """Select only topics data and save topics file into `/output_path/<keyword>.json`

    Args:
        scrape_result (list[dict]): raw result from API scraping (list of pagination responses)
        keyword (str): file name
        output_folder (str, optional): utput folder for the dataset. Defaults to ".".
    """

    topics = []

    for batch in scrape_result:
        if 'data' in batch.keys():
            topics += batch['data']

    if not os.path.exists(f"{output_folder}"):
        os.makedirs(f"{output_folder}", exist_ok=True)

    output_file = f"{output_folder}/{keyword}.json"

    with open(output_file, "w") as f:
        f.write(json.dumps(topics, ensure_ascii=False, indent=2))
        # print(output_file)
