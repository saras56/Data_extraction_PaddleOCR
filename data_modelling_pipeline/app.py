from __future__ import print_function
import joblib
import flask
import ast
import math
import pandas as pd
from io import StringIO

model_path = r'.\rf_model.joblib'

def extract_coordinates(df):
    df['x1_min'] = df['List A'].apply(lambda x: x[0])
    df['y1_min'] = df['List A'].apply(lambda x: x[1])
    df['x1_max'] = df['List A'].apply(lambda x: x[2])
    df['y1_max'] = df['List A'].apply(lambda x: x[3])
    
    df['x2_min'] = df['List B'].apply(lambda x: x[0])
    df['y2_min'] = df['List B'].apply(lambda x: x[1])
    df['x2_max'] = df['List B'].apply(lambda x: x[2])
    df['y2_max'] = df['List B'].apply(lambda x: x[3])
    
    return df

def calculate_dist(df):
    df['xmin_dist'] = abs(df['x2_min']-df['x1_min'])
    df['ymin_dist'] = abs(df['y2_min']-df['y1_min'])
    df['xmax_dist'] = abs(df['x2_max']-df['x1_max'])
    df['ymax_dist'] = abs(df['y2_max']-df['y1_max'])
    return df

def calculate_centroid(df):
    df['bb1_centroid_x'] = (df['x1_min']+df['x1_max'])/2
    df['bb1_centroid_y'] = (df['y1_min']+df['y1_max'])/2
    df['bb2_centroid_x'] = (df['x2_min']+df['x2_max'])/2
    df['bb2_centroid_y'] = (df['y2_min']+df['y2_max'])/2
    
    df['centroid_x_dist'] = abs(df['bb1_centroid_x']-df['bb2_centroid_x'])
    df['centroid_y_dist'] = abs(df['bb1_centroid_y']-df['bb2_centroid_y'])
    
    #distance between the two centroids
    df['centroid_dist'] = df['centroid_x_dist']**2+df['centroid_y_dist']**2
    df['centroid_dist'] = df.apply(lambda row:math.sqrt(row['centroid_dist']),axis =1)
    
    return df

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate areas of bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_iou_df(df):
    iou_values = []
    for _, row in df.iterrows():
        bbox1 = row[['x1_min', 'y1_min', 'x1_max', 'y1_max']]
        bbox2 = row[['x2_min', 'y2_min', 'x2_max', 'y2_max']]
        iou = calculate_iou(bbox1, bbox2)
        iou_values.append(iou)
    return iou_values

class ScoringService(object):
    model = None
    
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            with open(model_path, "rb") as inp:
                cls.model = joblib.load(inp)
        return cls.model
    

    @classmethod
    def predict(cls, processed, input_df):
        clf = cls.get_model()
        y_pred = clf.predict(processed)
        print(y_pred)
        mask = y_pred == 1
        result_df = input_df[mask]
        
        # Convert the DataFrame to a dictionary
        result_dict = result_df.to_dict(orient='records')
        
        return result_dict

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    health = ScoringService.get_model() is not None
    status = 200 if health else 404 # You can insert a health check here
    # status = 200
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    try:
        csv_data = flask.request.files["file"]
        df = pd.read_csv(StringIO(csv_data.read().decode('utf-8')))
        input_df = df.copy()
        input_df['List A'] = input_df['List A'].apply(lambda x: ast.literal_eval(x))
        input_df['List B'] = input_df['List B'].apply(lambda x: ast.literal_eval(x))
        cord_df = extract_coordinates(input_df)
        dist_df = calculate_dist(cord_df)
        centroid_df = calculate_centroid(dist_df)
        centroid_df['IOU'] = calculate_iou_df(centroid_df)
        centroid_df.drop(['List A','List B'],axis =1 ,inplace =True)
        
        res = ScoringService.predict(centroid_df,df)
            
        return {
            "status": "Success",
            "code": 200,
            "data": {
                "Sentiment" : res,
            },
            "message": "Prediction successfull"
        } 
        
    except Exception as e:
        error_message = str(e)
        error_code = 404  # File not found error

        return {
            "status": "Error",
            "code": error_code,
            "message": error_message
        }
        

