import boto3
import json
import tempfile
import io
from PIL import Image
import os
import numpy as np
import pickle

def fix_others(row, schema_set, threshold=0.7):
    """Convert predictions with low similarity to 'other' labels."""
    if row['results'] not in schema_set:
        return 'other'
    if row['score'] < threshold:
        return 'other'
    else: 
        return row['results']
    

def read_template_labels(bucket_name, key, s3resource):
    """Read labels corresponding to template matrix rows from s3."""
    content_object = s3resource.Object(bucket_name, key)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    data = json.loads(file_content)
    
    template_labels = list(data.keys())
    
    return template_labels

def load_pickle_from_s3(bucket, key, s3=boto3.resource('s3')):
    """Load pickle object from S3 bucket."""
    response = boto3.resource('s3').Bucket(bucket).Object(key).get()
    body_string = response['Body'].read()
    loaded_pickle = pickle.loads(body_string)
    return loaded_pickle

def strip_id(filename: str):
    """Strip main file tag from file path."""
    name = filename.split('.')[0]
    name = name.split('/')[-1]
    return name

def encoding_matrix(templates: dict)-> np.array:
    """"Convert template USE encodings into a N * 512 matrix."""
    use_matrix_list = []
    for key, value in templates.items():

        use_matrix_list.append(value)

    use_matrix = np.array(use_matrix_list).reshape((np.array(use_matrix_list).shape[0], 
                                                    np.array(use_matrix_list).shape[2])) 
    return use_matrix

def save_numpy_to_s3(matrix, s3client, bucket_name, key):
    """Store numpy array as a pickle object in S3."""
    array_data = io.BytesIO()
    pickle.dump(matrix, array_data)
    array_data.seek(0)
    s3client.upload_fileobj(array_data, bucket_name, key)

def json_from_image(bucket: str, bucket_name, directory: str, s3resource, s3client) -> dict:
    """
    Extract document text from image, split into individual words, store as a json.
    
        Parameters:
            bucket (s3 object): bucket where document images are stored.
            bucket_name (str): bucket where document images are stored.
            directory (str): directory where document images are stored.
            s3_resource (object): s3 resource.
            s3_client (object): s3 client.
        Returns:
            text_dict (dict): document names and extracted text.
    """
    textract = boto3.client('textract')
    text_dict = {}
    for object_summary in bucket.objects.filter(Prefix=directory):
        if object_summary.key.endswith('.png') or object_summary.key.endswith('.jpg'):

            document, image_bytes = s3_2_pil(BUCKET_NAME, object_summary.key)
            s3client.download_file(BUCKET_NAME, object_summary.key, f"{object_summary.key.split('/')[-1]}")

            width, height = document.size

            response = textract.detect_document_text(Document={'Bytes': image_bytes})

            id_n = 0
            forms = []
            for item in response["Blocks"]:

                if item["BlockType"] == "LINE":
                    for relationship in item['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            words_list = get_line_words(response, relationship['Ids'], width, height)

                    temp_dict = format_dict(item, words_list, width, height, id_n)
                    id_n += 1
                    forms.append(temp_dict)

            final_dict = {'doc_id':object_summary.key, 'form':forms, 'size':document.size}

            upload_path = object_summary.key.replace('template_images_current_cbp', 'template_jsons_cbp').replace('png','json').replace('jpg','json')
            
            text_dict[object_summary.key.split('/')[-1]] = text_from_response(response)
            
            s3object = s3resource.Object(BUCKET_NAME, upload_path)

            s3object.put(
                Body=(bytes(json.dumps(final_dict).encode('UTF-8')))
            )

            os.remove(f"{object_summary.key.split('/')[-1]}")
    
    return text_dict
            

def format_dict(item: dict, words: list, width: int, height: int, id_n: int) -> dict:
    """Return a formatted dict with required key-value pairs"""
    temp_dict = {}
    temp_dict['bbox'] = format_bbox(item['Geometry']['BoundingBox'], width, height)
    temp_dict['text'] = item['Text']
    temp_dict['words'] = words
    temp_dict['id'] = id_n
    
    return temp_dict

def s3_2_pil(bucket_name, key):
    """Read an image from s3 bucket into PIL format."""
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    obj = bucket.Object(key)

    tmp = tempfile.NamedTemporaryFile()

    with open(tmp.name, 'wb') as f:

        obj.download_fileobj(f)

        with open(tmp.name, 'rb') as document:
            image_bytes = bytearray(document.read())

        stream = io.BytesIO(image_bytes)
        pil_image=Image.open(stream)
   
    return pil_image, image_bytes


def get_line_words(response: dict, ids: list, width: int, height: int) -> list:
    """
    Return list of 'child' words contained in a textract "LINE" response. 
        
        Parameters:
            response (dict): textract response.
            ids (list): response relationship ids.
            width (int): document image dimension.
            height (int): document image dimension.
        Returns:
            words_list (List[str,...]): words in a sentence.
    """
    words_list = []
    for item in response["Blocks"]:
        if item["BlockType"] == "WORD" and item['Id'] in ids:
            words_list.append({"bbox": format_bbox(item['Geometry']['BoundingBox'],width, height), "text": item['Text']})
    return words_list


def format_bbox(geometry: dict, width: int, height: int) -> list:
    """
    Extracts normalized bboxes from textract geometry contained within response.
        Parameters:
            geometry (dict): 'Geometry' output from textract response.
            width (int): width of document image.
            height (int): height of document image.
        Returns: 
            actual_box (list): indices to corresponding entity bboxes.
    """
    
    left_scaled = int(geometry['Left']*width)
    width_scaled = int(geometry['Top']*height)
    top_scaled = int(geometry['Width']*width)
    height_scaled = int(geometry['Height']*height)
    box = [left_scaled, width_scaled, top_scaled, height_scaled]
    
    x, y, w, h = tuple(box) # the row comes in (left, top, width, height) format
    actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 
    return actual_box 

def text_from_response(response: dict)-> str:
    """Return a text string from all text detected by textract."""
    text = ""
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text = text + item["Text"] + " "
    return text

def text_from_response_no_handwriting(response: dict)-> str:
    """Return a text string from all text detected by textract with all handwriting removed."""
    text = ""
    for item in response["Blocks"]:
        if item["BlockType"] == "WORD" and item['TextType'] != 'HANDWRITING':
            
            text = text + item["Text"] + " "
    return text