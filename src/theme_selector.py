import argparse
import logging
import os
import shutil

from variants.variant_config import VariantConfig
from schema import VariantIdentifier

logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Selects theme based on the given arguments. Loads VariantConfig from variants/ directory and checks for existing variant."""
    parser = argparse.ArgumentParser(description="Theme selector for Streamlit app variants.")
    parser.add_argument("-a", "--app", required=True, type=str, help="Desired Streamlit app name")
    parser.add_argument("-v", "--variant", required=True, type=str, help="Variant name")
    args = parser.parse_args()

    logger.info(f"Copying theme for {args.app}/{args.variant}")

    config = VariantConfig(VariantIdentifier(streamlit_app_name=args.app, variant=args.variant))

    theme = config.get_or_fail("theme")

    theme_dir = os.path.abspath("themes/")
    theme_path = f"{theme_dir}/{theme}.toml"
    if not os.path.exists(theme_path):
        logger.warning("Theme file not found. Trying default.")
        theme_path = f"{theme_dir}/default.toml"
        if not os.path.exists(theme_path):
            logger.warning("Default theme file not found. Aborting.")
            exit()

    streamlit_dir = os.path.abspath(".streamlit/")
    if not os.path.exists(f"{streamlit_dir}"):
        os.makedirs(streamlit_dir)

    shutil.copyfile(theme_path, f"{streamlit_dir}/config.toml")
    logger.info(f"Streamlit config copied successfully from {theme_path}.")


if __name__ == "__main__":
    main()
