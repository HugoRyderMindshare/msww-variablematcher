"""
Remote (GCP) execution entrypoint for the matching Docker task.

Downloads target and candidate survey .sav files from GCS, runs the
matching pipeline, and uploads results to GCS.
"""

import argparse
import logging
import os
import resource
import sys
import time
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from google.cloud import storage
from pipeline import Pipeline, TaskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("match")


def log_resources(label: str) -> None:
    """Log current peak RSS memory usage."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if sys.platform == "linux":
        rss_mb /= 1024
    log.info("[%s] Peak RSS: %.1f MB", label, rss_mb)


load_dotenv("gcp.env")
load_dotenv("data.env")

DATA_BUCKET = os.environ["DATA_BUCKET"]


DATASET_DIR = "data/datasets/run"


def download_surveys(
    gcp_project: str,
    config_project: str,
    target_path: str,
    candidate_path: str,
) -> None:
    """Download target and candidate .sav files from separate GCS paths."""
    client = storage.Client(project=gcp_project)
    bkt = client.bucket(DATA_BUCKET)

    os.makedirs(DATASET_DIR, exist_ok=True)

    downloads = [
        (f"{config_project}/surveys/standardised/{target_path}.sav", "target.sav"),
        (f"{config_project}/surveys/standardised/{candidate_path}.sav", "candidate.sav"),
    ]

    for gcs_path, local_name in downloads:
        local_path = os.path.join(DATASET_DIR, local_name)
        log.info("Downloading gs://%s/%s ...", DATA_BUCKET, gcs_path)
        t0 = time.time()
        bkt.blob(gcs_path).download_to_filename(local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log.info(
            "Downloaded %.1f MB to %s (%.1fs)",
            size_mb,
            local_path,
            time.time() - t0,
        )


def upload_to_gcs(
    project: str,
    bucket: str,
    gcs_prefix: str,
    local_dir: str,
) -> None:
    """Upload all files in *local_dir* to GCS under *gcs_prefix*."""
    storage_client = storage.Client(project=project)
    storage_bucket = storage_client.bucket(bucket)

    file_count = 0
    for dirpath, _, filenames in os.walk(local_dir):
        for filename in filenames:
            local_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            blob = storage_bucket.blob(
                os.path.join(gcs_prefix, rel_path),
            )
            blob.upload_from_filename(local_path)
            file_count += 1
            log.info("  Uploaded %s", rel_path)

    log.info(
        "Uploaded %d file(s) to gs://%s/%s/", file_count, bucket, gcs_prefix
    )


if __name__ == "__main__":
    total_start = time.time()
    log.info("=" * 60)
    log.info("MATCHING PIPELINE — STARTING")
    log.info("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--config_project", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)

    args = parser.parse_args()

    log.info("Configuration:")
    log.info("  project:        %s", args.project)
    log.info("  config_project: %s", args.config_project)
    log.info("  target:         %s", args.target)
    log.info("  candidate:      %s", args.candidate)
    log.info("  timestamp:      %s", args.timestamp)
    log.info("  data_bucket:    %s", DATA_BUCKET)
    log_resources("startup")

    try:
        config = TaskConfig(dataset="run")

        # --- Download ---
        log.info("-" * 60)
        log.info("STEP 1/3: Downloading surveys from GCS")
        log.info("-" * 60)
        download_surveys(
            args.project, args.config_project, args.target, args.candidate,
        )
        log_resources("after download")

        with TemporaryDirectory() as temp_dir:
            # --- Pipeline ---
            log.info("-" * 60)
            log.info("STEP 2/3: Running matching pipeline")
            log.info("-" * 60)
            t0 = time.time()
            pipeline = Pipeline(temp_dir, config)
            pipeline.run()
            log.info("Pipeline completed in %.1fs", time.time() - t0)
            log_resources("after pipeline")

            # --- Upload results ---
            log.info("-" * 60)
            log.info("STEP 3/3: Uploading results to GCS")
            log.info("-" * 60)
            gcs_prefix = (
                f"{args.config_project}/surveys/matched/{args.timestamp}"
            )
            upload_to_gcs(
                project=args.project,
                bucket=DATA_BUCKET,
                gcs_prefix=gcs_prefix,
                local_dir=temp_dir,
            )

        log_resources("final")
        log.info("=" * 60)
        log.info(
            "PIPELINE COMPLETE — total time: %.1fs", time.time() - total_start
        )
        log.info("=" * 60)

    except Exception:
        log.exception("PIPELINE FAILED")
        sys.exit(1)
