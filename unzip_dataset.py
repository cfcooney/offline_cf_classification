import boto3
import pandas as pd
import zipfile
import configparser
from io import BytesIO
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s: %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def unzip_in_s3(bucket: str, prefix: str, save_key: str, s3=boto3.client('s3', use_ssl=False), s3_resource=boto3.resource('s3')) -> None:
    """
    Document dataset is stored as a zip file in s3. This function unzips it.
        Parameters:
            bucket (str): s3 bucket containing zip file.
            prefix (str): filepath to zip file.
            key (str): save path for unzipped document images.
            s3 (object): s3 client.
            s3_resource (object): s3 resource.
        Returns:
            None
    """
    
    zipped_keys =  s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter = "/")
    
    file_list = []
    for key in zipped_keys['Contents']:
        file_list.append(key['Key'])
            
    zip_obj = s3_resource.Object(bucket_name=bucket, key=file_list[0])
    buffer = BytesIO(zip_obj.get()["Body"].read())
    
    z = zipfile.ZipFile(buffer)
    for filename in z.namelist():
        file_info = z.getinfo(filename)
        s3_resource.meta.client.upload_fileobj(z.open(filename),
                                               Bucket=bucket,
                                               Key=f"{save_key}/{filename}")
        
if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    bucket_name = config['DatasetUnzip']['bucket_name']
    g_zip_key = config['DatasetUnzip']['prefix']
    save_path = config['DatasetUnzip']['save_path']
    excel_path = config['DatasetUnzip']['excel_path']
    s3_client = boto3.client('s3', use_ssl=False)
    s3_resource = boto3.resource('s3')
    
    logger.info("Unzipping dataset.")
    unzip_in_s3(bucket_name, g_zip_key, save_path, s3_client, s3_resource)
    logger.info("Unzipping complete.")
    
    logger.info("Saving filenames to csv.")
    bucket = s3_resource.Bucket(bucket_name)
    filename_list, label_list = [], []
    for n, object_summary in enumerate(bucket.objects.filter(Prefix=save_path)):

        if (object_summary.key.endswith('.png') or object_summary.key.endswith('.PNG')):     

            filename_list.append(object_summary.key)
            label_list.append(object_summary.key.split('/')[4])
    df = pd.DataFrame(data={'png_path' : filename_list, 'template' : label_list})
    
    df.to_excel(excel_path)
    logger.info("Filenames saved.")