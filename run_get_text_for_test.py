import pandas as pd
import json
import boto3
from utils import text_from_response, text_from_response_no_handwriting, s3_2_pil, strip_id
import configparser
import botocore


def main(filename, label, image_bucket_name, upload_bucket_name, upload_path, sub_directory, s3resource, textract, handwriting):
    """
    Run AWS Textract OCR on full test set to extract text and store in S3. Avoids recalling textract for each experiment.
        Parameters:
            filename (str): Universal Sentence Encoder
            label (str): data text
            image_bucket_name (str): s3 bucket
            upload_bucket_name (str): s3 bucket
            upload_path (str): s3 storage path
            sub_directory (str): directory
            s3resource (object): s3 resource
            textract (textract object): OCR engine
            handwriting (bool): Whether to include/exclude handwriting in classification
        Returns:
            None
    """
    
    upload_file = upload_path + sub_directory + strip_id(filename) + '.json'
 
    document, image_bytes = s3_2_pil(image_bucket_name, filename)
    
    try:
        response = textract.detect_document_text(Document={'Bytes': image_bytes})
        
        if HANDWRITING:
            text = text_from_response(response)
        else:
            text = text_from_response_no_handwriting(response)

        s3object = s3resource.Object(upload_bucket_name, upload_file)

        s3object.put(
            Body=(bytes(json.dumps({'text':text, 'label':label}).encode('UTF-8')))
        )
    except textract.exceptions.UnsupportedDocumentException as error:
        print(f"Unsupported document type for {filename}")
    

if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    UPLOAD_BUCKET_NAME = config['TextTest']['UPLOAD_BUCKET_NAME']
    IMAGE_BUCKET_NAME  = config['TextTest']['IMAGE_BUCKET_NAME']
    UPLOAD_PATH  = config['TextTest']['upload_path']
    dataset_path = config['TextTest']['DATASET']
    sub_directory = config['TextTest']['sub_directory']
    HANDWRITING = eval(config['TextTest']['HANDWRITING'])

    df = pd.read_excel(dataset_path, index_col=0, engine='openpyxl')
    
    filenames =  df['png_path'].values
    labels = df['template'].values
    s3resource = boto3.resource('s3')
    textract = boto3.client('textract')
    
    for filename, label in zip(filenames, labels):

        main(filename, label, IMAGE_BUCKET_NAME, UPLOAD_BUCKET_NAME, UPLOAD_PATH, sub_directory, s3resource, textract, HANDWRITING)