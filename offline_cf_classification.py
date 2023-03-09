import scipy.spatial as sp
import tensorflow_hub as hub
from utils import load_pickle_from_s3, read_template_labels
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
        format='%(asctime)s | %(name)s | %(levelname)s: %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def get_similar_form(form_vector, template_matrix):
    """Cosine simlarity applied to matrix-vector. Returns max, max index and similarity array."""
    similarity_array = 1 - sp.distance.cdist(form_vector, template_matrix, 'cosine')
    max_value = similarity_array.max()
    max_index = similarity_array.argmax()
    
    return max_value, max_index, similarity_array


def classify_offline_cf(text: str, model, use_matrix, template_labels):
    """"
    Classify document as an offline_cf or other. Text vector representation created using 
    universal sentence encoder. Cosine simularity is the distance metric against a matrix
    of templates. Returns document class.
    
        Parameters:
            text (str): Document text extracted from OCR.
            model (tensorflow model): Universal Sentence Encoder.
            use_matrix (numpy.array): USE encodings in matrix form.
            template_labels (list): labels corresponding to matrix row indices.
            
        Returns:
           document_class (str): predicted class.
           max_value (float): similarity score for predicted class.
    """
    
    vector = model([text])
    vector = vector.numpy().reshape((1, 512))

    max_value, max_index, _ = get_similar_form(vector, use_matrix)
    
    document_class = template_labels[max_index]
    return document_class, max_value


if __name__ ==' __main__':
    
    
    MODEL_URL = config['Templates']['MODEL_URL']
    model = hub.load(MODEL_URL)
    
    logger.info(f"Loading template matrix from: {BUCKET_NAME}/{MATRIX_PATH}")
    BUCKET_NAME = config['TextTest']['UPLOAD_BUCKET_NAME']
    MATRIX_DIR = config['Templates']['upload_path']
    MATRIX_PATH = MATRIX_DIR + config['Templates']['matrix_file']
    use_matrix = load_pickle_from_s3(BUCKET_NAME, MATRIX_PATH)
    
    template_labels = read_template_labels(BUCKET_NAME, JSON_PATH, s3resource)
    
    file = config['TextTest']['test_file']
    content_object = s3resource.Object(BUCKET_NAME, file)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    data = json.loads(file_content)
    text = data["text"]
    
    doc_class, score = classify_offline_cf(text, model, use_matrix, template_labels)
    
    logger.info(f"Document Class: {doc_class} | Score: {round(score,3)}")
    

    
    