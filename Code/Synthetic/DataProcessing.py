import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import tensorflow as tf

pd.set_option("display.max_columns", None)

height = 250
width = 250

# Global cache for loaded images
_image_cache = {}
_image_cache_vggface = {}


def loadData(col="Average", num_img=5500):
    if col == "Average":
        data = pd.read_csv("../../Data/ImageExp/Selected_Ratings.csv")
        data = data[["Filename", col]]
    else:
        data = pd.read_csv("../../Data/ImageExp/All_Ratings.csv")
        data = data[data["Rater"] == int(col)][["Filename", "Rating"]].rename(
            columns={"Filename": "Filename", "Rating": col}
        )
    data = data.sample(frac=1)
    data = data.head(num_img)

    return data


def retrievePixels(path):
    # Check cache first
    if path in _image_cache:
        return _image_cache[path]

    folder_path = "../../Data/Images/"
    img = tf.keras.utils.load_img(folder_path + path, target_size=(height, width))
    x = tf.keras.utils.img_to_array(img)

    # Cache the result
    _image_cache[path] = x
    return x


def retrievePixels_batch(paths):
    """Load multiple images in parallel"""
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(retrievePixels, paths))
    return results


def _preprocess_vggface(x):
    # Tutorial-compatible preprocessing: keep RGB and scale to [0, 1].
    return x.astype("float32") / 255.0


def retrievePixels_vggface(path):
    # Cache key must include target size / preprocessing variant.
    key = ("vggface_224", path)
    if key in _image_cache_vggface:
        return _image_cache_vggface[key]

    folder_path = "../../Data/Images/"
    img = tf.keras.utils.load_img(folder_path + path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = _preprocess_vggface(x)

    _image_cache_vggface[key] = x
    return x


def retrievePixels_batch_vggface(paths):
    """Load multiple images in parallel with tutorial-compatible preprocessing."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(retrievePixels_vggface, paths))
    return results


def _encode_from_filenames(files, idx, value):
    files = files.astype(str)
    return files.str[idx].eq(value).astype(int).tolist()


def _load_pixels_cached(series):
    arr = series.astype(str).tolist()
    return [retrievePixels(path) / 255.0 for path in arr]


def processData(h=250, w=250, col="Average", num_comp=1, num_img=5500):
    global height, width
    height = h
    width = w
    data = loadData(col=col, num_img=num_img)
    # threshold = data["Average"].describe()["std"]
    # threshold = round(threshold.item(), 3)

    train = data.sample(frac=0.8)
    test = data.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    protected_ts_sex = _encode_from_filenames(test["Filename"], 1, "M")
    protected_ts_race = _encode_from_filenames(test["Filename"], 0, "C")

    res_tr = []
    res_ts = []

    res_tr_single = []
    res_ts_single = []
    print("\nGenerating training data...")

    for indexA, rowA in train.iterrows():
        comp = []
        while len(comp) < num_comp:
            indexB = random.randint(0, len(train) - 1)
            rowB = train.iloc[indexB]
            if (indexA == indexB) or (indexB in comp):
                continue
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                res_tr.append(
                    {"A": rowA["Filename"], "B": rowB["Filename"], "Label": label}
                )

                res_tr_single.append(
                    {"A": rowA["Filename"], "B": rowB["Filename"], "Label": label}
                )

                res_tr.append(
                    {"A": rowB["Filename"], "B": rowA["Filename"], "Label": -label}
                )
                comp.append(indexB)
    data_tr = pd.DataFrame(res_tr)
    data_tr_single = pd.DataFrame(res_tr_single)

    data_tr["A"] = _load_pixels_cached(data_tr["A"])
    data_tr["B"] = _load_pixels_cached(data_tr["B"])

    data_tr_single["A"] = _load_pixels_cached(data_tr_single["A"])
    data_tr_single["B"] = _load_pixels_cached(data_tr_single["B"])
    # print("Saving training data...")
    # data_tr = data_tr.sample(frac=1)
    # data_tr.to_csv("../../Data/ImageExp/image_train.csv", index=False)
    print("Generating testing data...")
    for indexA, rowA in test.iterrows():
        comp = []
        # for indexB, rowB in test.iterrows():
        #     if (indexA == indexB) or (protected_ts[indexA] == protected_ts[indexB]):
        #         continue
        while len(comp) < num_comp:
            indexB = random.randint(0, len(test) - 1)
            rowB = test.iloc[indexB]
            if (indexA == indexB) or (indexB in comp):
                continue
            ratingA = rowA[col]
            ratingB = rowB[col]
            label = 0
            if ratingA > ratingB:
                label = 1
            elif ratingA < ratingB:
                label = -1
            if label != 0:
                res_ts.append(
                    {"A": rowA["Filename"], "B": rowB["Filename"], "Label": label}
                )
                res_ts_single.append(
                    {"A": rowA["Filename"], "B": rowB["Filename"], "Label": label}
                )
                res_ts.append(
                    {"A": rowB["Filename"], "B": rowA["Filename"], "Label": -label}
                )
                comp.append(indexB)
    data_ts = pd.DataFrame(res_ts)
    data_ts_single = pd.DataFrame(res_ts_single)

    protected_ts_A_sex = _encode_from_filenames(data_ts["A"], 1, "M")

    protected_ts_B_sex = _encode_from_filenames(data_ts["B"], 1, "M")

    protected_ts_A_race = _encode_from_filenames(data_ts["A"], 0, "C")

    protected_ts_B_race = _encode_from_filenames(data_ts["B"], 0, "C")

    protected_ts_AB_race = pd.DataFrame(
        {"A": protected_ts_A_race, "B": protected_ts_B_race}
    )

    protected_ts_AB_sex = pd.DataFrame(
        {"A": protected_ts_A_sex, "B": protected_ts_B_sex}
    )

    protected_ts_A_sex_single = _encode_from_filenames(data_ts_single["A"], 1, "M")

    protected_ts_B_sex_single = _encode_from_filenames(data_ts_single["B"], 1, "M")

    protected_ts_A_race_single = _encode_from_filenames(data_ts_single["A"], 0, "C")

    protected_ts_B_race_single = _encode_from_filenames(data_ts_single["B"], 0, "C")

    protected_ts_AB_race_single = pd.DataFrame(
        {"A": protected_ts_A_race_single, "B": protected_ts_B_race_single}
    )

    protected_ts_AB_sex_single = pd.DataFrame(
        {"A": protected_ts_A_sex_single, "B": protected_ts_B_sex_single}
    )

    data_ts["A"] = _load_pixels_cached(data_ts["A"])
    data_ts["B"] = _load_pixels_cached(data_ts["B"])

    data_ts_single["A"] = _load_pixels_cached(data_ts_single["A"])
    data_ts_single["B"] = _load_pixels_cached(data_ts_single["B"])

    # print("Saving testing data...")
    # data_ts = data_ts.sample(frac=1)
    # data_ts.to_csv("../../Data/ImageExp/image_test.csv", index=False)
    print("Done.")
    print("Training data size:", len(data_tr.index))
    print("Testing data size:", len(data_ts.index))
    print("Training data size:", len(data_tr_single.index))
    print("Testing data size:", len(data_ts_single.index))

    test_list = pd.DataFrame(
        [
            {"indexA": indexA, "A": rowA["Filename"], "Score": rowA[col]}
            for indexA, rowA in test.iterrows()
        ]
    )
    test_list["A"] = test_list["A"].apply(retrievePixels)

    data_list = pd.DataFrame(
        [
            {"indexA": indexA, "A": rowA["Filename"], "Score": rowA[col]}
            for indexA, rowA in data.iterrows()
        ]
    )
    data_list["A"] = data_list["A"].apply(retrievePixels)

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
        protected_ts_AB_race,
        protected_ts_AB_sex,
        protected_ts_AB_race_single,
        protected_ts_AB_sex_single,
    )
