
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

# Run for a specific font
# CMD: make font=MY_FONT generate_w_font
.PHONY: generate_w_font
generate_w_font:
	docker compose exec image-generator trdg -c 10 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -k 90 -rk -al 1 -im L -ft /app/trdg/fonts/en/$(font)

# Generate massive combinaison of all fonts
# CMD: make run
generate_all:
	docker compose exec image-generator trdg -c 11700 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -k 90 -rk -al 1 -im L -fd /app/trdg/fonts/en/

# Generate image for each font in specific folder
# CMD: make generate_grouped
FONTS_DIR = ./fonts/en
generate_grouped: $(FONTS_DIR)/*
	for file in $^; do\
		font=`basename $${file}`;\
		echo "[STARTING] $${font}";\
		filename=$${font%%.*};\
		echo "    Creating folder $${filename}";\
		mkdir -p "./output/$${filename}"\
		echo "    Created";\
		echo "    Starting image generation";\
		docker compose exec image-generator trdg -c 10 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/$${filename} -e png -k 90 -rk -al 1 -im L -ft /app/trdg/fonts/en/$${font};\
		echo "    End image generation";\
		echo "[END] $${font}";\
	done

# Delete all the generated images
generate_delete:
	rm -rf ./output/*.png
