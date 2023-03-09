import configparser
import boto3
from utils import json_from_image, encoding_matrix, save_numpy_to_s3
import tensorflow_hub as hub
import json
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s: %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def generate_templates(model, data, template_dict={}) -> dict:
    """
    Generate new templates and add to existing template dict.
        
        Parameters:
            model (tensorflow model): Universal Sentence Encoder
            data (dict): data text
            template_dict (dict): existing templates
        Returns:
            template_dict (dict): Revised templates dictionary with new templates added.
    """
    
    for key, value in data.items():
        
        word_embeddings = model([value])
        template_dict[key.split('.')[0]] = word_embeddings.numpy().tolist()
    
    return template_dict

def main(bucket, bucket_name, directory, model, s3resource, s3client, upload_path, json_path, matrix_path):
    """
    Create templates from document images and store as a pickle file.
    
    Parameters:
            bucket (s3 bucket): s3 bucket object
            bucket_name (str): bucket name
            directory (str): template images directory
            model (tensorflow model): Universal Sentence Encoder
            s3resource (object): s3 resource
            s3client (object): s3 client
            upload_path (str): path to templates
            json_path (str): path to upload json
            matrix_path (str): path to upload matrix
        Returns:
            None
    """
    
    template_text = json_from_image(bucket, bucket_name, directory, s3resource, s3client)

    template_dict = generate_templates(model, template_text)
    
    s3object = s3resource.Object(BUCKET_NAME, json_path)

    s3object.put(
        Body=(bytes(json.dumps(template_dict).encode('UTF-8')))
    )
    
    template_matrix =  encoding_matrix(template_dict)
    
    save_numpy_to_s3(template_matrix, s3client, bucket_name, matrix_path)


if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
   
    BUCKET_NAME = config['Templates']['BUCKET_NAME']
    DIRECTORY = config['Templates']['directory']
    UPLOAD_PATH = config['Templates']['upload_path']
    JSON_PATH = UPLOAD_PATH + config['Templates']['templates_file']
    MATRIX_PATH = UPLOAD_PATH + config['Templates']['matrix_file']
    
    s3resource = boto3.resource('s3')
    s3client = boto3.client('s3')
    BUCKET = s3resource.Bucket(BUCKET_NAME)
    
    MODEL_URL = config['Templates']['MODEL_URL']
    model = hub.load(MODEL_URL)
    
    logger.info(f"Beginning creation of templates")
    main(BUCKET, BUCKET_NAME, DIRECTORY, model, s3resource, s3client, UPLOAD_PATH, JSON_PATH, MATRIX_PATH)
    logger.info(f"Template creation complete")
    
    

