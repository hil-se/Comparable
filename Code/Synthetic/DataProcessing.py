import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

pd.set_option("display.max_columns", None)

height = 250
width = 250
MAX_IMAGE_WORKERS = 8

_image_cache = {}
_image_cache_vggface = {}
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "Data"


def loadData(col="Average", num_img=5500):
    if col == "Average":
        data = pd.read_csv(DATA_DIR / "ImageExp" / "Selected_Ratings.csv")
        data = data[["Filename", col]]
    else:
        data = pd.read_csv(DATA_DIR / "ImageExp" / "All_Ratings.csv")
        data = data[data["Rater"] == int(col)][["Filename", "Rating"]].rename(
            columns={"Filename": "Filename", "Rating": col}
        )
    data = data.sample(frac=1)
    data = data.head(num_img)

    return data


def _load_cached_image(path, target_size, cache, *, cache_key=None, preprocess=None):
    cache_key = path if cache_key is None else cache_key
    if cache_key in cache:
        return cache[cache_key]

    folder_path = DATA_DIR / "Images"
    image = tf.keras.utils.load_img(str(folder_path / path), target_size=target_size)
    pixels = tf.keras.utils.img_to_array(image)
    if preprocess is not None:
        pixels = preprocess(pixels)
    cache[cache_key] = pixels
    return pixels


def retrievePixels(path):
    return _load_cached_image(path, (height, width), _image_cache)


def retrievePixels_batch(paths):
    with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
        return list(executor.map(retrievePixels, paths))


def _preprocess_vggface(x):
    # Tutorial-compatible preprocessing: keep RGB and scale to [0, 1].
    return x.astype("float32") / 255.0


def retrievePixels_vggface(path):
    return _load_cached_image(
        path,
        (224, 224),
        _image_cache_vggface,
        cache_key=("vggface_224", path),
        preprocess=_preprocess_vggface,
    )


def retrievePixels_batch_vggface(paths):
    with ThreadPoolExecutor(max_workers=MAX_IMAGE_WORKERS) as executor:
        return list(executor.map(retrievePixels_vggface, paths))


def _encode_from_filenames(files, idx, value):
    files = files.astype(str)
    return files.str[idx].eq(value).astype(int).tolist()


def _load_pixels_cached(series):
    arr = series.astype(str).tolist()
    return [retrievePixels(path) / 255.0 for path in arr]


def _pair_label(score_a, score_b):
    return int(np.sign(score_a - score_b))


def _build_pair_frames(frame, score_col, num_comp):
    rows = frame[["Filename", score_col]].to_dict("records")
    pair_rows = []
    single_rows = []

    for idx_a, row_a in enumerate(rows):
        partners = set()
        while len(partners) < num_comp:
            idx_b = random.randrange(len(rows))
            if idx_b == idx_a or idx_b in partners:
                continue

            row_b = rows[idx_b]
            label = _pair_label(row_a[score_col], row_b[score_col])
            if label == 0:
                continue

            pair = {"A": row_a["Filename"], "B": row_b["Filename"], "Label": label}
            pair_rows.extend([pair, {"A": pair["B"], "B": pair["A"], "Label": -label}])
            single_rows.append(pair)
            partners.add(idx_b)

    return pd.DataFrame(pair_rows), pd.DataFrame(single_rows)


def _protected_pair_frame(df):
    return {
        "race": pd.DataFrame(
            {
                "A": _encode_from_filenames(df["A"], 0, "C"),
                "B": _encode_from_filenames(df["B"], 0, "C"),
            }
        ),
        "sex": pd.DataFrame(
            {
                "A": _encode_from_filenames(df["A"], 1, "M"),
                "B": _encode_from_filenames(df["B"], 1, "M"),
            }
        ),
    }


def _attach_pair_pixels(df):
    df["A"] = _load_pixels_cached(df["A"])
    df["B"] = _load_pixels_cached(df["B"])
    return df


def _score_frame(df, score_col):
    scored = pd.DataFrame(
        {
            "indexA": np.arange(len(df)),
            "A": df["Filename"].values,
            "Score": df[score_col].values,
        }
    )
    scored["A"] = scored["A"].apply(retrievePixels)
    return scored


def processData(h=250, w=250, col="Average", num_comp=1, num_img=5500):
    global height, width
    height = h
    width = w
    data = loadData(col=col, num_img=num_img)

    train = data.sample(frac=0.8)
    test = data.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    protected_ts_sex = _encode_from_filenames(test["Filename"], 1, "M")
    protected_ts_race = _encode_from_filenames(test["Filename"], 0, "C")

    print("\nGenerating training data...")
    data_tr, data_tr_single = _build_pair_frames(train, col, num_comp)
    data_tr = _attach_pair_pixels(data_tr)
    data_tr_single = _attach_pair_pixels(data_tr_single)

    print("Generating testing data...")
    data_ts, data_ts_single = _build_pair_frames(test, col, num_comp)
    protected_ts = _protected_pair_frame(data_ts)
    protected_ts_single = _protected_pair_frame(data_ts_single)
    data_ts = _attach_pair_pixels(data_ts)
    data_ts_single = _attach_pair_pixels(data_ts_single)

    print("Done.")
    print("Training data size:", len(data_tr.index))
    print("Testing data size:", len(data_ts.index))
    print("Training data size:", len(data_tr_single.index))
    print("Testing data size:", len(data_ts_single.index))

    test_list = _score_frame(test, col)
    data_list = _score_frame(data, col)

    return (
        data_tr,
        data_ts,
        data_tr_single,
        data_ts_single,
        test_list,
        data_list,
        len(data_tr.index),
        len(data_ts.index),
        len(data_tr_single.index),
        len(data_ts_single.index),
        train,
        test,
        protected_ts_race,
        protected_ts_sex,
        protected_ts["race"],
        protected_ts["sex"],
        protected_ts_single["race"],
        protected_ts_single["sex"],
    )
