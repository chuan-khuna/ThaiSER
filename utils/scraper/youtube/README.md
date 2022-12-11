In this version of Youtube scraper, it needs to use `selenium` for getting the first `continuation` token, `payload` and `header`.

Todo

- [ ] Use only `requests`?

```py
# initialise scraper
scraper = YouTubeScraper("./chromedriver")
scraper.selenium_scroll = 6
scraper.selenium_wait = 2
scraper.max_lazyload = 40

url = "https://www.youtube.com/watch?v=SNdf5oSVYDc"
# list of lazy load responses
data = scraper.scrape(url)

comments = []
for res in data:
    lazyload_comments = extract_comments_from_response(res)
    comments += lazyload_comments
```
