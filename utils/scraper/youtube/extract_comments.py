import re
import json
import pandas as pd


def extract_comments_from_response(json_obj: dict) -> list[dict]:
    """return all comments(str) and authors in a lazy load response
    """

    extracted_comments = []

    try:
        if "onResponseReceivedEndpoints" in json_obj.keys():
            for continue_items in json_obj["onResponseReceivedEndpoints"]:
                # select key to dig deeper
                if "reloadContinuationItemsCommand" in continue_items.keys():
                    k = "reloadContinuationItemsCommand"
                elif "appendContinuationItemsAction" in continue_items.keys():
                    k = "appendContinuationItemsAction"
                # comments are in this key's list
                for item in continue_items[k]["continuationItems"]:
                    if "commentThreadRenderer" in item.keys():
                        comment_obj = item["commentThreadRenderer"]["comment"]

                        author_name = comment_obj["commentRenderer"]["authorText"]["simpleText"]
                        comment_texts = comment_obj["commentRenderer"]["contentText"]["runs"]

                        # concatenate text
                        concat_text = ""
                        for text in comment_texts:
                            concat_text += text["text"]

                        comment = {"author": author_name, "text": concat_text}

                        extracted_comments.append(comment)
    except:
        pass

    return extracted_comments