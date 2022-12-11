import json
import yaml
import requests
import re
from .twitter_scraper_base import TwitterScraperBase


class TweetScraper(TwitterScraperBase):

    def __init__(self, header_yaml_path: str):
        self.max_lazyload = 10
        self.api_url = "https://twitter.com/i/api/2/search/adaptive.json"
        self.headers = self._load_header_from_yaml(header_yaml_path)
        self.data_to_extract = ['tweets', 'users']
        # default twitter search option is searching for top tweets
        # it will reach lazyload limit ~20 pages (not sure)
        self.search_latest_tweets = False

    def _build_payload(self, keyword: str, cursor_token: str) -> dict:
        # see more: https://requests.readthedocs.io/en/latest/user/quickstart/
        # Passing Parameters In URLs (GET request)
        # you can find the configuration by inspecting payload on twitter search page

        params = {
            'include_profile_interstitial_type':
                1,
            'include_blocking':
                1,
            'include_blocked_by':
                1,
            'include_followed_by':
                1,
            'include_want_retweets':
                1,
            'include_mute_edge':
                1,
            'include_can_dm':
                1,
            'include_can_media_tag':
                1,
            'include_ext_has_nft_avatar':
                1,
            'include_ext_is_blue_verified':
                1,
            'skip_status':
                1,
            'cards_platform':
                'Web-12',
            'include_cards':
                1,
            'include_ext_alt_text':
                True,
            'include_ext_limited_action_results':
                False,
            'include_quote_count':
                True,
            'include_reply_count':
                1,
            'tweet_mode':
                'extended',
            'include_ext_collab_control':
                True,
            'include_entities':
                True,
            'include_user_entities':
                True,
            'include_ext_media_color':
                True,
            'include_ext_media_availability':
                True,
            'include_ext_sensitive_media_warning':
                True,
            'include_ext_trusted_friends_metadata':
                True,
            'send_error_codes':
                True,
            'simple_quoted_tweet':
                True,
            # keyword to search here
            'q':
                'sawano hiroyuki',
            'count':
                20,
            'query_source':
                'typed_query',
            'pc':
                1,
            'spelling_corrections':
                1,
            'include_ext_edit_control':
                True,
            'ext':
                'mediaStats,highlightedLabel,hasNftAvatar,voiceInfo,enrichments,superFollowMetadata,unmentionInfo,editControl,collab_control,vibe'
        }

        params['q'] = keyword

        if cursor_token:
            params['cursor'] = cursor_token

        if self.search_latest_tweets == True:
            params['tweet_search_mode'] = 'live'

        return params

    def _extract_token(self, response: dict) -> str | None:
        """Extract cursor token from a raw response
            TODO: refactor this spaghetti code

            Args:
                response (dict): raw response

            Returns:
                str | None: a string of cursor token, None if not exists
            """

        try:
            if 'timeline' in response.keys():
                timeline_data = response['timeline']
                if 'instructions' in timeline_data.keys():
                    instructions_data = timeline_data['instructions']

            all_entries = []

            # flatten entry data
            for item in instructions_data:
                for k in item.keys():
                    if k == 'addEntries':
                        entries_data = item['addEntries']['entries']
                    elif k == 'replaceEntry':
                        entries_data = item['replaceEntry']['entry']

                    if isinstance(entries_data, list):
                        all_entries += entries_data
                    else:
                        all_entries += [entries_data]

            # find cursor token
            for entry in all_entries:
                if 'operation' in entry['content'].keys():
                    operation = entry['content']['operation']
                    if 'cursor' in operation.keys():
                        cursor = operation['cursor']
                        if cursor['cursorType'] == 'Bottom':
                            cursor_token = cursor['value']
        except Exception:
            print("Cannot find cursor token")
            cursor_token = None

        return cursor_token

    def _flatten_dict(self, dict_: dict) -> list[dict]:
        return list(dict_.values())

    def process_response(self, response: dict) -> dict[str, list[dict]]:
        """Process raw response to extract only interested information

        Args:
            response (dict): raw response

        Returns:
            dict[list[dict]]: a dictionary where its keys is the information to extract, 
            eg `tweets`, `users`. Its values are a list of dicts of that entity.

            example {'tweets': [{...}, {...}], 'users': [{...}, {...}]}

            return {} if key not found
        """

        processed_data = {}
        if 'globalObjects' in response.keys():
            data = response['globalObjects']

            for k in self.data_to_extract:
                if k in data.keys():
                    processed_data[k] = self._flatten_dict(data[k])
                else:
                    processed_data[k] = []

        return processed_data

    def scrape_lazyload(self, keyword: str, cursor_token: str) -> dict:
        """Request for a lazyload response, return {} if error occurs

        Args:
            keyword (str): _description_
            cursor_token (str): _description_

        Returns:
            dict: raw json response
        """

        params = self._build_payload(keyword, cursor_token)

        try:
            res = requests.get(self.api_url, params=params, headers=self.headers, timeout=5)

            if res.status_code == 200:
                response_json = res.json()
            else:
                response_json = {}
        except Exception:
            response_json = {}

        return response_json

    def scrape(self, keyword: str) -> dict[str, list[dict]]:
        """Scrape for tweets that contain a keyword from twitter (search bar)

        Args:
            keyword (str): _description_

        Returns:
            dict[str, list[dict]]: a dictionary where its keys is the information to extract, 
            eg `tweets`, `users`. Its values are a list of dicts of that entity.

            example {'tweets': [{...}, {...}], 'users': [{...}, {...}]}
        """

        # initialise cleaned data
        data = {}
        for k in self.data_to_extract:
            data[k] = []

        cursor_token = ''
        for i in range(1, self.max_lazyload + 1):
            print(f"Lazyload page: {i}")

            # request and process data
            res = self.scrape_lazyload(keyword, cursor_token)
            processed_data = self.process_response(res)
            cursor_token = self._extract_token(res)

            blank_response = True
            print("- Found", end=" ")
            for k in self.data_to_extract:
                if k in processed_data.keys():
                    data[k] += processed_data[k]
                    # if there is data in processed_data -> set flat to not break this loop
                    blank_response = blank_response and (len(processed_data[k]) == 0)
                    print(f"{k}: {len(processed_data[k])}", end=' ')
            print("")
            # check condition to break the loop
            # if 'globalObjects' not in res.keys():
            #     break
            if cursor_token is None:
                break
            if blank_response:
                break
        return data