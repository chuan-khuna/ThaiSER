import os
import re
import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def preprocess_topic_file(topic_json_file: str) -> pd.DataFrame:
    """Get topic json file path
    preprocess only interested columns
    return DataFrame

    Args:
        topic_json_file (str): path/to/topics.json

    Returns:
        pd.DataFrame: topic dataframe
    """
    topic_cols = ['id', 'url', 'created_time', 'title', 'detail', 'rooms']
    df = pd.read_json(topic_json_file)
    df = df[topic_cols]
    df['type'] = 'topic'
    df = df.rename(columns={'id': 'topic_id'})
    return df


def _preprocess_comments(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Select only interested columns from comments data
    """
    comment_cols = ['_id', 'comment_no', 'data_utime', 'message']
    df = raw_df[comment_cols]
    df['type'] = 'comment'
    df = df.rename(columns={'_id': 'comment_id'})
    return df


def _preprocess_replies(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Select only interested columns from replies data
    """

    # filter only comment that has replies
    raw_replies_df = raw_df[raw_df['replies'].apply(len) > 0]
    reply_cols = ['reply_id', 'reply_no', 'comment_no', 'message', 'data_utime']

    # preprocess json object in a column
    # list of cleaned df
    replies_normalized = []
    for raw_reply in raw_replies_df['replies']:
        replies_normalized.append(pd.json_normalize(raw_reply))

    # preprocess columns
    # if no replies then return a blank dataframe
    if len(replies_normalized) > 0:
        replies_df = pd.concat(replies_normalized)
        replies_df = replies_df[reply_cols]
        replies_df['type'] = 'reply'
    else:
        replies_df = pd.DataFrame()
    return replies_df


def preprocess_comments_file(comments_file: str) -> pd.DataFrame:
    """Get a file that stores comments data
    Preprocess and return dataframe

    Args:
        comments_file (str): path/to/comments_file (<topic_id>.json)

    Returns:
        pd.DataFrame: comments-replies dataframe
    """
    topic_id = comments_file.strip(".json").split("/")[-1]

    # comments json for a topic
    with open(comments_file) as f:
        raw_json = json.load(f)

    raw_df = pd.json_normalize(raw_json)
    comments_df = _preprocess_comments(raw_df)
    replies_df = _preprocess_replies(raw_df)

    cleaned_df = pd.concat([comments_df, replies_df])
    cleaned_df['topic_id'] = topic_id
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df


def preprocess_keyword_dataset(topic_file: str, comments_path: str) -> pd.DataFrame:
    """Preprocess the dataset of a scraped keyword
    ie create a `.csv` dataset for scraped json files

    Args:
        topic_file (str): path/to/topic_file.json (keyword.json)
        comments_path (str): path/to/comments_folder (raw_comments)

    Returns:
        pd.DataFrame: cleaned dataframe that contains texts from topics, comments, replies
    """

    topic_df = preprocess_topic_file(topic_file)

    cleaned_comments_dfs = []
    for f in os.listdir(comments_path):
        file_path = comments_path + f
        try:
            comment_reply_df = preprocess_comments_file(file_path)
            cleaned_comments_dfs.append(comment_reply_df)
        except:
            pass

    df = pd.concat([topic_df, pd.concat(cleaned_comments_dfs)])
    df = df.reset_index(drop=True)

    return df


TEXT_COLS = ['title', 'detail', 'message']


def filter_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """Filter only rows in the dataframe that contian keyword

    return: dataframe without index columns/without resetting index
    the index of this dataframe will be used to map to the raw dataframe
    """

    df[TEXT_COLS] = df[TEXT_COLS].fillna('')

    filtered_dataframes = []
    for col in TEXT_COLS:
        filtered_df = df[df[col].str.contains(keyword)]
        filtered_dataframes.append(filtered_df)

    filtered_df = pd.concat(filtered_dataframes).reset_index().drop_duplicates('index').drop(
        columns=['index'])
    return filtered_df


def melt_dataframe(df: pd.DataFrame):
    """melt/unpivot dataframe into format id, text_type, text
    Transfom the dataset `df` from wide-form into long-form of text columns
    """
    # the index column is used for joining melted dataframe with raw dataframe
    melted = df[TEXT_COLS].reset_index().melt(id_vars='index', value_name='text', var_name='type')
    # drop blank text
    melted = melted[melted['text'].apply(len) > 0]
    melted = melted.reset_index(drop=True)
    return melted