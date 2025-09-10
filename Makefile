COMPOSE=infra/docker-compose.yml

.PHONY: up down build logs ps reindex restart clean

up:
	@docker compose -f $(COMPOSE) up -d redis ml-bridge rag gateway advisor report market ui

down:
	@docker compose -f $(COMPOSE) down

build:
	@docker compose -f $(COMPOSE) build

logs:
	@docker compose -f $(COMPOSE) logs -f

ps:
	@docker compose -f $(COMPOSE) ps

reindex:
	@docker compose -f $(COMPOSE) run --rm \
		-e GENAI_API_KEY -e EMBEDDING_MODEL -e CHROMA_DIR -e COLLECTION_NAME \
		rag python build_index.py

restart:
	@docker compose -f $(COMPOSE) restart redis ml-bridge rag gateway advisor report market ui

clean:
	@docker compose -f $(COMPOSE) down -v

