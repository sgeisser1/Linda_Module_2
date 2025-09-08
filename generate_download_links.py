import boto3
from botocore.config import Config
import numpy as np

def generate_presigned_url(bucket_name, object_name, expiration=86400, region="eu-central-1"):
    try:
        s3_client = boto3.client(
            "s3",
            region_name=region,
            config=Config(s3={'addressing_style': 'virtual'}, signature_version='s3v4')
        )

        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )
        return url
    except:
        return "Credentials not available"


if __name__ == "__main__":
    links = []
    for year in np.arange(2010, 2025):
        links.append(generate_presigned_url("km-hydrology-cas-aml-1", f"aare_{year}.csv"))
    
    with open("links.txt", "w") as f:
        for item in links:
            f.write(f"{item}\n")