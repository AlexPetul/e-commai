run:
	docker compose up --build api

dev:
	PYTHONPATH=src uv run chainlit run src/ydachnik_chatbot/chat.py --port 8000

scrape:
	uv run scrapy crawl catalog -s JOBDIR=.scrapy-job

scrape-fresh:
	rm -rf .scrapy-job products.csv && uv run scrapy crawl catalog -s JOBDIR=.scrapy-job

scrape-test:
	uv run scrapy crawl catalog -s CLOSESPIDER_ITEMCOUNT=10 -o /tmp/test.csv

loaddata:
	docker compose run --build --rm loaddata

recreate-metadata:
	docker compose run --build --rm loaddata --recreate-metadata
