from utils import strip_id
import json
import pickle
import boto3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp
import pandas as pd
import configparser
import json
from utils import s3_2_pil, text_from_response, load_pickle_from_s3, read_template_labels
import tensorflow_hub as hub
from botocore.exceptions import ClientError
from offline_cf_classification import get_similar_form, classify_offline_cf
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s: %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )



def main(filenames, upload_path, sub_directory, upload_bucket_name, use_matrix, template_labels):
    """
    Compare uploaded claim form with each of the template forms using cosine similarity with matrix multiplication.
    Returns a document class and the predicted probability.
        Parameters:
            filenames (list): list of claim for filepaths.
            upload_path (str): path within s3 bucket.
            sub_directory (str): path within s3 bucket.
            upload_bucket_name (str): labels corresponding to matrix row indices.
            use_matrix (numpy.array): USE encodings in matrix form.
            template_labels (list): labels corresponding to matrix row indices.
        Returns:
            results (list): list of class labels.
            score (list): list of similarity scores.
    """
    
    results, score = [], []

    for filename in filenames:
        
        file = upload_path + sub_directory + strip_id(filename) + '.json'
       
        try:
            content_object = s3resource.Object(upload_bucket_name, file)

            file_content = content_object.get()['Body'].read().decode('utf-8')

            data = json.loads(file_content)

            text = data["text"]
            label = data["label"]

            predicted_class, probability = classify_offline_cf(text, model, use_matrix, template_labels)

            results.append(predicted_class)
            score.append(probability)

        except ClientError as ce:
            if ce.response['Error']['Code'] == 'NoSuchKey':
                results.append(-100)
                score.append(-100)
                
    return results, score
          
                
if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    dataset_path = config['TextTest']['DATASET']
    sub_directory = config['TextTest']['sub_directory']
    BUCKET_NAME = config['TextTest']['UPLOAD_BUCKET_NAME']
    UPLOAD_PATH  = config['TextTest']['upload_path']
    
    MATRIX_DIR = config['Templates']['upload_path']
    MATRIX_PATH = MATRIX_DIR + config['Templates']['matrix_file']
    JSON_PATH = MATRIX_DIR + config['Templates']['templates_file']

    s3resource = boto3.resource('s3')
    textract = boto3.client('textract')
    
    MODEL_URL = config['Templates']['MODEL_URL']
    model = hub.load(MODEL_URL)
    
    df = pd.read_excel(dataset_path, index_col=0, engine='openpyxl')
    
    filenames =  df['png_path'].values
    logger.info(f"Loading template matrix from: {BUCKET_NAME}/{MATRIX_PATH}")
    use_matrix = load_pickle_from_s3(BUCKET_NAME, MATRIX_PATH)
    logger.info(f"Loading template labels from: {BUCKET_NAME}/{JSON_PATH}")
    template_labels = read_template_labels(BUCKET_NAME, JSON_PATH, s3resource)
    
    logger.info(f"Beginning classification run on test set: {UPLOAD_PATH + sub_directory}")
    results, score = main(filenames, UPLOAD_PATH, sub_directory, BUCKET_NAME, use_matrix, template_labels)
    logger.info(f"Completed classification run")
    
    SAVE_PATH  = config['TextTest']['save_path']
    df["results"] = results
    df["score"] = score
    df.to_excel(SAVE_PATH)
    
    logger.info(f"Results saved to: {SAVE_PATH}")

    