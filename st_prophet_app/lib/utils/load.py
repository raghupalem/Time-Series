from typing import Any, Dict, Tuple

import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import toml
from PIL import Image


def get_project_root() -> str:
    """Returns project root path.

    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent.parent)


@st.cache_data(ttl=300)
def load_custom_config(config_file: io.BytesIO) -> Dict[Any, Any]:
    """Loads config toml file from user's file system as a dictionary.

    Parameters
    ----------
    config_file
        Uploaded toml config file.

    Returns
    -------
    dict
        Loaded config dictionary.
    """
    toml_file = Path(get_project_root()) / f"config/custom_{config_file.name}"
    write_bytesio_to_file(str(toml_file), config_file)
    config = toml.load(toml_file)
    return dict(config)


def write_bytesio_to_file(filename: str, bytesio: io.BytesIO) -> None:
    """
    Write the contents of the given BytesIO to a file.

    Parameters
    ----------
    filename
        Uploaded toml config file.
    bytesio
        BytesIO object.
    """
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


@st.cache_data(ttl=300)
def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(Path(get_project_root()) / f"references/{image_name}")

@st.cache_data(ttl=300)
def load_config(
    config_streamlit_filename: str, config_instructions_filename: str, config_readme_filename: str
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Loads configuration files.

    Parameters
    ----------
    config_streamlit_filename : str
        Filename of lib configuration file.
    config_instructions_filename : str
        Filename of custom config instruction file.
    config_readme_filename : str
        Filename of readme configuration file.

    Returns
    -------
    dict
        Lib configuration file.
    dict
        Readme configuration file.
    """
    config_streamlit = toml.load(Path(get_project_root()) / f"config/{config_streamlit_filename}")
    config_instructions = toml.load(
        Path(get_project_root()) / f"config/{config_instructions_filename}"
    )
    config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
    return dict(config_streamlit), dict(config_instructions), dict(config_readme)
