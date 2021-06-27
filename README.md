# Data-Centric AI Competition

[Data-Centric AI Competition](https://https-deeplearning-ai.github.io/data-centric-comp/)

## Contest

[Data-Centric AI Competition Submission Guide](https://worksheets.codalab.org/worksheets/0x7a8721f11e61436e93ac8f76da83f0e6)

## Submission

[zilongqiu-ai-competition CodaLab](https://worksheets.codalab.org/worksheets/0x5e0056b28ee94cf6bd3175657601fc54)

Submit a zip file containing all the images following this structure below:

```
sample_submission/
    train/
        i/
        ii/
        iii/
        iv/
        ...
    val/
        i/
        ii/
        iii/
        iv/
        ...
```

## Run training

1. Verify the `train.py` script folders path (`directory`, `user_data`, `valid_data` & `test_data`)
2. *(optional)* if using `testing` folder, verify the images
3. `cd scripts`
4. `python3 train.py`

## Generate images

1. `cd scripts/generator`
2. *(optional)* Verify the dictionary data in the `dicts/en.txt` file
3. *(optional)* Verify the fonts in the `fonts/en` directory
4. `make up` or `make rebuild`
5. Makefile available commands:
    - Generate images with a specific font with `make font=MY_FONT generate_w_font`
    - Generate massive combinaison of all fonts `make generate_all`
    - Generate image for each font in specific folder `make generate_grouped`
    - Delete all the generated images `make generate_delete`
    - Copy all the generated images to testing folder `make generate_to_testing`
6. *(optional)* If using `make generate_all`, you need to split the `testing/train` images to the `testing/val` folder (`train` 80% / `val` 20%)
