from os import path
from typing import Any

import yaml

from schema import VariantIdentifier


class VariantConfig:
    config: dict[str, Any] | None = None

    def __init__(self, identifier: VariantIdentifier):
        self.identifier = identifier
        self.config = self.get_config()

    def get(self, key: str, default: Any) -> Any:
        """Returns a config value by key or default value provided"""
        config = self.get_config()
        return config.get(key, default)

    def get_or_fail(self, key: str) -> Any:
        """Returns a config value by key or fails if key not provided"""
        # TODO: may be expanded with type definition
        config = self.get_config()

        if key not in config:
            raise ValueError(
                f"Configuration {key} missing in {self.identifier['streamlit_app_name']}/{self.identifier['variant']}"
            )

        return config[key]

    def get_config(self) -> dict[str, Any]:
        if self.config is not None:
            return self.config

        identifier = self.identifier
        base_path = (
            f"{path.dirname(path.abspath(__file__))}/{identifier['streamlit_app_name'].lower()}"
        )

        identifier["variant"] = "" if identifier["variant"] is None else identifier["variant"]

        if path.exists(f"{base_path}/{identifier['variant']}.yml"):
            filename = identifier["variant"]
        elif path.exists(f"{base_path}/default.yml"):
            filename = "default"
        else:
            raise Exception("Failed to load variant config.")

        with open(f"{base_path}/{filename}.yml") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise TypeError("Config file must contain a dictionary at the top level")

        return config
