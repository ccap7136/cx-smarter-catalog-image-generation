import argparse
import vertexai

from data_utils import *
from tqdm import tqdm
from vertexai.batch_prediction import BatchPredictionJob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time



def run_batch_prediction_job(
        input_file,
        output_uri=GCS["GCS_OUTPUT_URI"],
        model_id=LLM_MODELS["NANO_BANANA"],
        project_id=GCS["PROJECT_ID"],
        location=GCS["LOCATION"],
):
    """
    Run a batch prediction job using the Vertex AI API.

    Args:
        input_file (str): Path to the input JSONL file (GCS URI)
        output_uri (str): GCS URI for the output
        model_id (str): Model ID to use for prediction
        project_id (str): GCP project ID
        location (str): GCP region

    Returns:
        BatchPredictionJob: The batch prediction job
    """
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    print(f"\nSubmitting batch prediction job with model: {model_id}")
    print(f"Input file: {input_file}")
    print(f"Output URI: {output_uri}")

    # Submit the batch prediction job
    batch_prediction_job = BatchPredictionJob.submit(
        source_model=model_id,
        input_dataset=input_file,
        output_uri_prefix=output_uri,
    )

    # Print job information
    print(f"Job resource name: {batch_prediction_job.resource_name}")
    print(f"Model resource name with the job: {batch_prediction_job.model_name}")
    print(f"Job state: {batch_prediction_job.state.name}")

    return batch_prediction_job

def monitor_batch_prediction_job(batch_prediction_job, poll_interval=30):
    """
    Monitor a batch prediction job until it completes.

    Args:
        batch_prediction_job (BatchPredictionJob): The batch prediction job to monitor
        poll_interval (int): Interval in seconds to poll for job status

    Returns:
        str: Output location of the job
    """
    print(f"Monitoring batch prediction job: {batch_prediction_job.resource_name}")

    # Refresh the job until complete
    while not batch_prediction_job.has_ended:
        print(f"Job state: {batch_prediction_job.state.name}")
        time.sleep(poll_interval)
        batch_prediction_job.refresh()

    # Check if the job succeeds
    if batch_prediction_job.has_succeeded:
        print("Job succeeded!")
    else:
        print(f"Job failed: {batch_prediction_job.error}")

    # Check the location of the output
    print(f"Job output location: {batch_prediction_job.output_location}")
    return batch_prediction_job.output_location


def main():
    # Set up argument parser similar to gdsn_extraction.py but with added batch options
    parser = argparse.ArgumentParser(
        description='Generate lifestyle images for selected products using reference images.')
    parser.add_argument('--gcs_uri_input', type=str, default=None,
                        help='GCS URI for image generation input file')
    parser.add_argument('--batch_job', type=str, default='test',
                        help='Name of batch job')

    args = parser.parse_args()
    batch_job = args.batch_job
    os.makedirs(batch_job, exist_ok=True)

    # Image Generation
    try:
        job = run_batch_prediction_job(
            input_file=args.gcs_uri_input,
            output_uri=GCS["GCS_OUTPUT_URI"],
            model_id=LLM_MODELS["NANO_BANANA"],
            project_id=GCS["PROJECT_ID"],
            location=GCS["LOCATION"],
        )

        # Monitor the batch prediction job
        print("Monitoring batch prediction job...")
        img_gen_output_location = monitor_batch_prediction_job(job)
    except Exception as e:
        print(f"Error in batch prediction workflow: {e}")

    # Download and process output if requested
    if job.has_succeeded and img_gen_output_location:
        img_gen_output_location = "gs://sysco-smarter-catalog-ce-image-generation-poc/batch_outputs/prediction-model-2026-03-02T18:25:04.588422Z"
        print(f"Downloading image generation response from {img_gen_output_location}...")
        output_dir = download_gcs_output(
            img_gen_output_location,
            local_dir=f"./{batch_job}",
            project_id=GCS["PROJECT_ID"]
        )

if __name__ == "__main__":
    main()