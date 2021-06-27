
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

# Run for a specific font
# CMD: make font=MY_FONT generate_w_font
.PHONY: generate_w_font
generate_w_font:
	docker compose exec image-generator trdg -c 10 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -k 90 -rk -al 1 -im L -wd 32 -ft /app/trdg/fonts/en/$(font)

# Generate massive combinaison of all fonts
# CMD: make run
generate_all:
	docker compose exec image-generator trdg -c 9000 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/ -e png -k 90 -rk -al 1 -im L -wd 32 -b 1 -fd /app/trdg/fonts/en/

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
		docker compose exec image-generator trdg -c 10 -t 2 -dt /app/trdg/dicts/en.txt --output_dir /app/trdg/output/out/$${filename} -e png -k 90 -rk -al 1 -im L -wd 32 -ft /app/trdg/fonts/en/$${font};\
		echo "    End image generation";\
		echo "[END] $${font}";\
	done

# Delete all the generated images
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

# Copy all the generated images to training folder
generate_to_training:
	rm -rf ./../../testing/train/i/*.png
	rm -rf ./../../testing/train/ii/*.png
	rm -rf ./../../testing/train/iii/*.png
	rm -rf ./../../testing/train/iv/*.png
	rm -rf ./../../testing/train/v/*.png
	rm -rf ./../../testing/train/vi/*.png
	rm -rf ./../../testing/train/vii/*.png
	rm -rf ./../../testing/train/viii/*.png
	rm -rf ./../../testing/train/ix/*.png
	rm -rf ./../../testing/train/x/*.png
	cp ./output/I_*.png ./../../testing/train/i/
	cp ./output/II_*.png ./../../testing/train/ii/
	cp ./output/III_*.png ./../../testing/train/iii/
	cp ./output/IV_*.png ./../../testing/train/iv/
	cp ./output/V_*.png ./../../testing/train/v/
	cp ./output/VI_*.png ./../../testing/train/vi/
	cp ./output/VII_*.png ./../../testing/train/vii/
	cp ./output/VIII_*.png ./../../testing/train/viii/
	cp ./output/IX_*.png ./../../testing/train/ix/
	cp ./output/X_*.png ./../../testing/train/x/