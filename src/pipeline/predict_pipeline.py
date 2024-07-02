import sys
import pandas as pd
from src.exception import CustomeException
from src.utils import load_object


class PredictPipelines():
    def __init__(self):
        pass # by default empty constructor
    
    def predict(self,features):
        try:        
            model_path = 'artifact/model.pkl'
            preprocessor_path  = 'artifact/processor.pkl'
            model = load_object(file_path=model_path)  # load object function only import the pkl and just load the pkl file 
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features) # when we load the data, first we need to do is to scale the data
            preds = model.predict(data_scaled) # after tranforming the data, our model will just predict the data
            return preds # returning the predidcitons
        except Exception as e:
            raise CustomeException(e,sys)
        


# Custom Data class will responsible for mapping all the inputs that we are giving in the html to the backend with a particular values.
class CustomData():
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        # now creating variable using self, and these value will come from web application
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    # this function will return all of my above mentioned input in a dataframe 
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomeException(e, sys)
        

