from pathlib import Path

DL_CONFIG_PATH     =    Path(__file__).resolve().parent
DL_SOURCE_PATH     =    Path(DL_CONFIG_PATH).resolve().parent
DL_BASE_PATH       =    Path(DL_SOURCE_PATH).resolve().parent
DL_DATA_PATH       =    DL_BASE_PATH    /   "data"
DL_IMG_DATA_PATH   =    DL_BASE_PATH    /   "data"    / "image"