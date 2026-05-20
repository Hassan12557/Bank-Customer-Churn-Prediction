.PHONY: dev down clean

# The default action when you just type 'make'
dev:
	docker compose down
	docker compose up --build -d

# To safely turn off the ecosystem
down:
	docker compose down

# Clear all cached docker elements to save disk space
clean:
	docker compose down --rmi all --volumes --remove-orphans