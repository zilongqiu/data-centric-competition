
# Auto train
.PHONY: auto_train
auto_train:
	rm -rf ../data/model/*
	touch ../data/model/results.txt
	#rm -rf ../data/train/*
	#rm -rf ../data/val/*
	#cp -r ../best/train/* ../data/train/
	#cp -r ../best/val/* ../data/val/
	cd generator && $(MAKE) generate_all
	python3 train.py
	sleep 5
	echo "[START] Renaming generated training data"
	@mmv ../data/train/\*/I_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/II_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/III_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/IV_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/V_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VI_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VII_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VIII_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/IX_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/X_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	echo "[END] Renaming generated training data"
	sleep 5
	sh ../best/best.sh
	sleep 15
	$(MAKE) auto_train

# Auto train by cumulating the generated images
.PHONY: auto_train_cumulative
auto_train_cumulative:
	rm -rf ../data/model/*
	touch ../data/model/results.txt
	rm -rf ../data/train/*
	rm -rf ../data/val/*
	cp -r ../best/train/* ../data/train/
	cp -r ../best/val/* ../data/val/
	cd generator && $(MAKE) generate_all
	python3 train.py
	sleep 5
	echo "[START] Renaming generated training data"
	@mmv ../data/train/\*/I_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/II_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/III_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/IV_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/V_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VI_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VII_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VIII_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/IX_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/X_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	echo "[END] Renaming generated training data"
	sleep 5
	sh ../best/best.sh
	sleep 15
	$(MAKE) auto_train

# Auto train with data_augmentation using imgaug
.PHONY: auto_train_imgaug
auto_train_imgaug:
	rm -rf ../data/model/*
	touch ../data/model/results.txt
	rm -rf ../data/train/*
	rm -rf ../data/val/*
	cp -r ../best/train/* ../data/train/
	cp -r ../best/val/* ../data/val/
	cd generator/data_augmentation && $(MAKE) clean
	cd generator/data_augmentation && $(MAKE) run
	cd generator/data_augmentation && $(MAKE) output_to_train
	python3 train.py
	sleep 5
	echo "[START] Renaming generated training data"
	@mmv ../data/train/\*/I_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/II_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/III_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/IV_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/V_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VI_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VII_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/VIII_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/IX_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	@mmv ../data/train/\*/X_\* ../data/train/\#1/generated_$$(openssl rand -hex 12)_\#2
	echo "[END] Renaming generated training data"
	sleep 5
	sh ../best/best.sh
	sleep 15
	$(MAKE) auto_train