import json
import yaml
import requests
import re
from .twitter_scraper_base import TwitterScraperBase


class ThreadScraper(TwitterScraperBase):

    def __init__(self, header_yaml_path: str):
        self.max_lazyload = 10
        self.api_url = "https://twitter.com/i/api/graphql/BoHLKeBvibdYDiJON1oqTg/TweetDetail"
        self.headers = self._load_header_from_yaml(header_yaml_path)

    def _build_payload(self, tweet_id: str, cursor_token: str) -> dict:
        params = {
            "variables": {
                "focalTweetId": "",
                "referrer": "tweet",
                "with_rux_injections": False,
                "includePromotedContent": True,
                "withCommunity": True,
                "withQuickPromoteEligibilityTweetFields": True,
                "withBirdwatchNotes": False,
                "withSuperFollowsUserFields": True,
                "withDownvotePerspective": True,
                "withReactionsMetadata": False,
                "withReactionsPerspective": False,
                "withSuperFollowsTweetFields": True,
                "withVoice": True,
                "withV2Timeline": True
            },
            "features": {
                "responsive_web_twitter_blue_verified_badge_is_enabled": True,
                "verified_phone_label_enabled": False,
                "responsive_web_graphql_timeline_navigation_enabled": True,
                "unified_cards_ad_metadata_container_dynamic_card_content_query_enabled": True,
                "tweetypie_unmention_optimization_enabled": True,
                "responsive_web_uc_gql_enabled": True,
                "vibe_api_enabled": True,
                "responsive_web_edit_tweet_api_enabled": True,
                "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
                "standardized_nudges_misinfo": True,
                "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": False,
                "interactive_text_enabled": True,
                "responsive_web_text_conversations_enabled": False,
                "responsive_web_enhance_cards_enabled": True
            }
        }

        params["variables"]["focalTweetId"] = tweet_id

        if cursor_token:
            params["variables"]["cursor"] = cursor_token

        return params

    def _extract_token(self, response: dict) -> str:
        tokens = []
        try:
            # get thread data; time line add
            thread_data = response['data']['threaded_conversation_with_injections_v2'][
                'instructions'][0]

            for entry in thread_data['entries']:
                try:
                    token = entry['content']['itemContent']['value']
                    tokens.append(token)
                except Exception:
                    pass

        except KeyError:
            pass

        if len(tokens) > 0:
            token = tokens[0]
        else:
            token = None
        return token

    def __extract_replies_from_entry(self, entry: dict) -> list[dict]:
        # an entry -- a reply grop the reply of a tweet and its replies(if it has)
        # so we need to extract 1. reply of a tweet + 2. reply of a reply

        entry_replies = []
        try:
            entry_content = entry['content']

            # loop through a group of reply
            # to extract the reply ifself and its replies
            for item in entry_content['items']:
                reply_detail = item['item']['itemContent']['tweet_results']['result']['legacy']
                entry_replies.append(reply_detail)

        except Exception:
            pass

        return entry_replies

    def process_response(self, response: dict) -> list[dict]:
        replies_in_response = []
        try:
            # thread data (tweet and its replies) -- a group of replies in a lazyload
            raw_thread_data = response['data']['threaded_conversation_with_injections_v2'][
                'instructions'][0]
            for entry in raw_thread_data['entries']:
                replies = self.__extract_replies_from_entry(entry)
                if len(replies) > 0:
                    replies_in_response += replies
        except Exception:
            pass

        return replies_in_response

    def scrape_lazyload(self, tweet_id: str, cursor_token: str) -> dict:
        """
        ref: to use nested dictionary as query params https://stackoverflow.com/questions/48193316/passing-a-nested-dictionary-to-requests-module

        Args:
            tweet_id (str): _description_
            cursor_token (str): _description_

        Returns:
            dict: _description_
        """

        params = self._build_payload(tweet_id, cursor_token)

        try:
            res = requests.get(self.api_url, json=params, headers=self.headers, timeout=5)
            if res.status_code == 200:
                response_json = res.json()
            else:
                response_json = {}
        except Exception:
            response_json = {}

        return response_json

    def scrape(self, tweet_id: str) -> list[dict]:
        all_replies = []
        cursor_token = ''
        for i in range(1, self.max_lazyload + 1):
            print(f"Lazyload page: {i}")
            res = self.scrape_lazyload(tweet_id, cursor_token)
            processed_response = self.process_response(res)
            cursor_token = self._extract_token(res)
            print(f"- Found {len(processed_response)} replies")
            all_replies += processed_response

            if cursor_token is None:
                break
            if len(processed_response) == 0:
                break
        return all_replies