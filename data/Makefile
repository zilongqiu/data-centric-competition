
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
