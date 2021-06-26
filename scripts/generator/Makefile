
.PHONY: list help
list help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

.PHONY: up
up:
	$(MAKE) down
	docker compose up -d
	docker compose run image-generator

.PHONY: rebuild
rebuild:
	$(MAKE) down
	docker compose up -d --force-recreate --build
	docker compose run image-generator

.PHONY: down
down:
	docker compose down

.PHONY: logs
logs:
	docker compose logs -f

.PHONY: enter
enter:
	docker compose run image-generator

run:
	docker compose exec image-generator trdg -c 10 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -k 90 -rk -al 1 -im L -ft /app/trdg/fonts/en/$(font)

FONTS_DIR = ./fonts/en
DOCKER_OUTPUT_DIR = /app/trdg/output/out
l: $(FONTS_DIR)/*
	for file in $^; do\
		font=`basename $${file}`;\
		echo "[STARTING] $${font}";\
		filename=$${font%%.*};\
		echo "    Creating folder $${filename}";\
		mkdir -p "./output/$${filename}"\
		echo "    Created";\
		echo "    Starting image generation";\
		docker compose exec image-generator trdg -c 10 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/$${filename} -e png -k 90 -rk -al 1 -im L -ft /app/trdg/fonts/en/$${font};\
		echo "    End image generation";\
		echo "[END] $${font}";\
	done

del:
	rm -rf ./output/*.jpg