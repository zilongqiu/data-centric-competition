
.PHONY: list help
list help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Delete train & val images
.PHONY: delete_all
delete_all:
	$(MAKE) delete_train
	$(MAKE) delete_val

# Delete train images
.PHONY: delete_train
delete_train:
	rm -rf ./train/i/*.png
	rm -rf ./train/ii/*.png
	rm -rf ./train/iii/*.png
	rm -rf ./train/iv/*.png
	rm -rf ./train/v/*.png
	rm -rf ./train/vi/*.png
	rm -rf ./train/vii/*.png
	rm -rf ./train/viii/*.png
	rm -rf ./train/ix/*.png
	rm -rf ./train/x/*.png

# Delete val images
.PHONY: delete_val
delete_val:
	rm -rf ./val/i/*.png
	rm -rf ./val/ii/*.png
	rm -rf ./val/iii/*.png
	rm -rf ./val/iv/*.png
	rm -rf ./val/v/*.png
	rm -rf ./val/vi/*.png
	rm -rf ./val/vii/*.png
	rm -rf ./val/viii/*.png
	rm -rf ./val/ix/*.png
	rm -rf ./val/x/*.png

# Copy images from results version
.PHONY: copy_version
copy_version:
	$(MAKE) delete_all
	cp -r ../results/$(VERSION)/train/* ./train/
	cp -r ../results/$(VERSION)/val/* ./val/

.PHONY: remove_number_prefix
remove_number_prefix:
	mmv ./train/i/I_\*.png ./train/i/\#1.png
	mmv ./train/ii/II_\*.png ./train/ii/\#1.png
	mmv ./train/iii/III_\*.png ./train/iii/\#1.png
	mmv ./train/iv/IV_\*.png ./train/iv/\#1.png
	mmv ./train/v/V_\*.png ./train/v/\#1.png
	mmv ./train/vi/VI_\*.png ./train/vi/\#1.png
	mmv ./train/vii/VII_\*.png ./train/vii/\#1.png
	mmv ./train/viii/VIII_\*.png ./train/viii/\#1.png
	mmv ./train/ix/IX_\*.png ./train/ix/\#1.png
	mmv ./train/x/X_\*.png ./train/x/\#1.png
