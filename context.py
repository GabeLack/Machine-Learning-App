import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ModelContext:
    def __init__(self,
                 #data_file_path: str,
                 df: pd.DataFrame,
                 target_column: str,
                 test_size: float = 0.3,
                 is_pipeline: bool = True,
                 scaler=StandardScaler()):
        #self.data_file_path = data_file_path
        self.target_column = target_column
        self.test_size = test_size
        self.is_pipeline = is_pipeline
        self.scaler = scaler
        self.df = df
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=101)

    def check_missing(self):
        missing_values = self.df.isna().sum()
        for column, count in missing_values.items():
            if count > 0:
                raise UserWarning(f"Warning: Column {column} has {count} missing values.")

    def check_feature_types(self):
        object_columns = self.X.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            raise UserWarning(f"Warning: Columns {list(object_columns)} are of object type and may need encoding.")
