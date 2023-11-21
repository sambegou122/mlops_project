
import pandas as pd
import numpy as np


class ModelManager():

    def __init__(self, model, text, file, output_file) -> None:
        self.model = model
        self.text = text
        self.file = file
        self.output_file = output_file


    def predict(self)-> None:
        """Predict the sentiment of the input text."""

        if self.text is not None:
            # df = pd.DataFrame(data = {"review": self.text}, index = [0])
            df = [self.text]
        else:
            df = pd.read_csv(self.file)
            assert list(df.columns) == ['review'], f"Expected columns: ['review'], but got {df.columns}"

        predictions = self.model.predict(df)

        if self.text:
            return predictions[0]
        else:
            df['prediction'] = predictions
            df.to_csv(self.output_file, index=False)
            print(f"Predictions saved to {self.output_file}")
            return None

    