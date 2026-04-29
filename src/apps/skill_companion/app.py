import asyncio
import os

from schema import VariantIdentifier
from variants.variant_config import VariantConfig

from Skill_Companion import main


def run() -> None:
    config = VariantConfig(
        VariantIdentifier(
            streamlit_app_name="Skill_Companion",
            variant=os.getenv("VARIANT", "default"),
        )
    )
    asyncio.run(main(config))


if __name__ == "__main__":
    run()
