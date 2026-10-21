# Data

Put the Yelp files here (or pass another folder with `--data-dir`). They are not committed:
the Yelp Open Dataset license doesn't allow redistribution, and the files total ~3 GB.

Download: https://www.yelp.com/dataset

| File                | Required | Contents                                   |
|---------------------|----------|--------------------------------------------|
| `yelp_train.csv`    | yes      | `user_id,business_id,stars` training ratings |
| `user.json`         | yes      | user profiles                              |
| `business.json`     | yes      | business metadata                          |
| `yelp_val.csv`      | no       | held-out ratings, used only for evaluation |
| `review_train.json` | no       | review votes and text                      |
| `checkin.json`      | no       | check-in counts                            |
| `tip.json`          | no       | tips                                       |
| `photo.json`        | no       | photo metadata                             |
