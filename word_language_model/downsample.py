import sys
import warnings
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append("/home/herman/git/muspy/")
import muspy

DATASET_DIR = Path("/data4/herman/muspy-new")
TARGET_DIR = DATASET_DIR / "downsampled"

DATASET_KEYS = [
    "nes",
    "jsb",
    "maestro",
    "hymnal",
    "hymnal_tune",
    "music21",
    "music21jsb",
    "nmd",
    "essen",
    "lmd",
    "wikifonia",
]


def get_dataset(key):
    if key == "lmd":
        return muspy.LakhMIDIDataset(DATASET_DIR / "lmd")
    if key == "wikifonia":
        return muspy.WikifoniaDataset(DATASET_DIR / "wikifonia")
    if key == "nes":
        return muspy.NESMusicDatabase(DATASET_DIR / "nes")
    if key == "jsb":
        return muspy.JSBChoralesDataset(DATASET_DIR / "jsb")
    if key == "maestro":
        return muspy.MAESTRODatasetV2(DATASET_DIR / "maestro")
    if key == "hymnal":
        return muspy.HymnalDataset(DATASET_DIR / "hymnal")
    if key == "hymnal_tune":
        return muspy.HymnalTuneDataset(DATASET_DIR / "hymnal_tune")
    if key == "music21":
        return muspy.MusicDataset(DATASET_DIR / "music21")
    if key == "music21jsb":
        return muspy.MusicDataset(DATASET_DIR / "music21jsb")
    if key == "nmd":
        return muspy.NottinghamDatabase(DATASET_DIR / "nmd")
    if key == "essen":
        return muspy.EssenFolkSongDatabase(DATASET_DIR / "essen")
    raise ValueError("Unrecognized dataset name.")


def _downsampler(d_key, music, idx, n_digits):
    prefix = "0" * (n_digits - len(str(idx)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            music.adjust_resolution(4)
            music.save_json(TARGET_DIR / d_key / (prefix + str(idx) + ".json"))
    except Exception:  # pylint: disable=broad-except
        return False
    return True


def process(d_key):
    print("Start processing dataset: {}".format(d_key))
    dataset = get_dataset(d_key)
    (TARGET_DIR / d_key).mkdir(exist_ok=True)
    n_digits = len(str(len(dataset)))

    results = Parallel(n_jobs=20, verbose=5)(
        delayed(_downsampler)(d_key, music, idx, n_digits)
        for idx, music in enumerate(dataset)
    )
    count = results.count(True)

    # count = 0
    # for idx in tqdm(range(len(dataset))):
    #     prefix = "0" * (n_digits - len(str(idx)))
    #     try:
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             music = dataset[idx]
    #             music.adjust_resolution(4)
    #             music.save_json(
    #                 TARGET_DIR / d_key / (prefix + str(idx) + ".json")
    #             )
    #         count += 1
    #     except Exception:  # pylint: disable=broad-except
    #         continue

    print("{} out of {} files successfully saved.".format(count, len(dataset)))


if __name__ == "__main__":
    if sys.argv[1].lower() == "all":
        for dataset_key in DATASET_KEYS:
            process(dataset_key)
    else:
        process(sys.argv[1].lower())
