
.PHONY: list help
list help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

.PHONY: build
build:
	sh ./script/build

.PHONY: run
run:
	sh ./script/run

.PHONY: clean
clean:
	rm -rf ./output/
	mkdir ./output

.PHONY: clean_temp
clean_temp:
	rm -rf ./temp/*/*.png

.PHONY: rename_temp_to_input
rename_temp_to_input:
	@mmv ./temp/i/\*.png ./input/I_\#1.png
	@mmv ./temp/ii/\*.png ./input/II_\#1.png
	@mmv ./temp/iii/\*.png ./input/III_\#1.png
	@mmv ./temp/iv/\*.png ./input/IV_\#1.png
	@mmv ./temp/v/\*.png ./input/V_\#1.png
	@mmv ./temp/vi/\*.png ./input/VI_\#1.png
	@mmv ./temp/vii/\*.png ./input/VII_\#1.png
	@mmv ./temp/viii/\*.png ./input/VIII_\#1.png
	@mmv ./temp/ix/\*.png ./input/IX_\#1.png
	@mmv ./temp/x/\*.png ./input/X_\#1.png

.PHONY: temp_to_input
temp_to_input:
	cp ./temp/*/*.png ./input

.PHONY: output_to_train
output_to_train:
	cp ./output/I_*.png ./../../../data/train/i/
	cp ./output/II_*.png ./../../../data/train/ii/
	cp ./output/III_*.png ./../../../data/train/iii/
	cp ./output/IV_*.png ./../../../data/train/iv/
	cp ./output/V_*.png ./../../../data/train/v/
	cp ./output/VI_*.png ./../../../data/train/vi/
	cp ./output/VII_*.png ./../../../data/train/vii/
	cp ./output/VIII_*.png ./../../../data/train/viii/
	cp ./output/IX_*.png ./../../../data/train/ix/
	cp ./output/X_*.png ./../../../data/train/x/
