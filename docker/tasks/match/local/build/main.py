"""
Local execution entrypoint for the matching Docker task.

Loads configuration from ``config.json`` and runs the shared
matching pipeline, writing outputs to the output directory.
"""

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from pipeline import Pipeline, TaskConfig

load_dotenv("/app/.env")


def main(config_path: str = "config.json") -> None:
    config = TaskConfig.from_json(config_path)
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"/app/output/{timestamp}"
    pipeline = Pipeline(output_dir=output_dir, config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
