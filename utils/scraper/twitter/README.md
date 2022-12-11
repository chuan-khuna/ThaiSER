# How to use tweets scraper

This scraper will gather tweets from Twitter search

```py
scraper = Tweetscraper('path/to/header.yml')
scraper.max_lazyload = 20

# list of tweets (dict)
data = scraper.scrape('Sawano Hiroyuki')
```

A YAML file that contains **request header** params
it should contain (log in search) `['authorization', 'cookie', 'x-csrf-token']` (nov 2022)
for incognito search it also need `'x-guest-token'`

```py
scraper = ThreadScraper("header.yml")

# return a list of replies of tweet=tweet_id
scraper.scrape("tweet_id")
```

# Note

- [ ] DRY this code
