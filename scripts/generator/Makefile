
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

.PHONY: enter
enter:
	docker compose run image-generator

# Generate images with a specific font
# CMD: make font=MY_FONT generate_w_font
.PHONY: generate_w_font
generate_w_font:
	$(MAKE) generate_delete
	docker compose exec image-generator trdg -c 10 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -k 90 -rk -al 1 -im L -wd 32 -ft /app/trdg/fonts/en/$(font)

# Generate massive combinaison of all fonts
# CMD: make generate_all
.PHONY: generate_all
generate_all:
	$(MAKE) generate_delete
	docker compose exec image-generator trdg -c 1000 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -hw -al 1 -im L -wd 32 -b 1 -fd /app/trdg/fonts/en/
	$(MAKE) generate_to_training

# Generate image for each font in specific folder
# CMD: make generate_grouped
FONTS_DIR = ./fonts/en
.PHONY: generate_grouped
generate_grouped: $(FONTS_DIR)/*
	rm -rf ./output/*
	for file in $^; do\
		font=`basename $${file}`;\
		echo "[STARTING] $${font}";\
		filename=$${font%%.*};\
		echo "    Creating folder $${filename}";\
		mkdir -p "./output/$${filename}"\
		echo "    Created";\
		echo "    Starting image generation";\
		docker compose exec image-generator trdg -c 10 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/$${filename} -e png -k 90 -rk -al 1 -im L -wd 32 -ft /app/trdg/fonts/en/$${font};\
		echo "    End image generation";\
		echo "[END] $${font}";\
	done

# Delete all the generated images
.PHONY: generate_delete
generate_delete:
	rm -rf ./output/I_*.png
	rm -rf ./output/II_*.png
	rm -rf ./output/III_*.png
	rm -rf ./output/IV_*.png
	rm -rf ./output/V_*.png
	rm -rf ./output/VI_*.png
	rm -rf ./output/VII_*.png
	rm -rf ./output/VIII_*.png
	rm -rf ./output/IX_*.png
	rm -rf ./output/X_*.png
	rm -rf ./output/*

# Copy all the generated images to training folder
.PHONY: generate_to_training
generate_to_training:
	rm -rf ./../../data/train/i/*.png
	rm -rf ./../../data/train/ii/*.png
	rm -rf ./../../data/train/iii/*.png
	rm -rf ./../../data/train/iv/*.png
	rm -rf ./../../data/train/v/*.png
	rm -rf ./../../data/train/vi/*.png
	rm -rf ./../../data/train/vii/*.png
	rm -rf ./../../data/train/viii/*.png
	rm -rf ./../../data/train/ix/*.png
	rm -rf ./../../data/train/x/*.png
	rm -rf ./../../data/val/i/*.png
	rm -rf ./../../data/val/ii/*.png
	rm -rf ./../../data/val/iii/*.png
	rm -rf ./../../data/val/iv/*.png
	rm -rf ./../../data/val/v/*.png
	rm -rf ./../../data/val/vi/*.png
	rm -rf ./../../data/val/vii/*.png
	rm -rf ./../../data/val/viii/*.png
	rm -rf ./../../data/val/ix/*.png
	rm -rf ./../../data/val/x/*.png
	cp ./output/I_*.png ./../../data/train/i/
	cp ./output/II_*.png ./../../data/train/ii/
	cp ./output/III_*.png ./../../data/train/iii/
	cp ./output/IV_*.png ./../../data/train/iv/
	cp ./output/V_*.png ./../../data/train/v/
	cp ./output/VI_*.png ./../../data/train/vi/
	cp ./output/VII_*.png ./../../data/train/vii/
	cp ./output/VIII_*.png ./../../data/train/viii/
	cp ./output/IX_*.png ./../../data/train/ix/
	cp ./output/X_*.png ./../../data/train/x/