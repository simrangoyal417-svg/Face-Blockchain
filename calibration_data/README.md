# Face calibration dataset

Place labeled photos in identity directories:

```text
calibration_data/
  same_person/
    person_a/
      image_1.jpg
      image_2.jpg
    person_b/
      image_1.jpg
      image_2.jpg
  different_people/
    person_c/
      image_1.jpg
      image_2.jpg
    person_d/
      image_1.jpg
      image_2.jpg
```

Images in the same identity directory are labeled `same_person`. Images from
different identity directories are labeled `different_person`. Use consented,
representative photos with one clearly visible face per image. This directory
is intended for local evaluation data and should not contain API keys.