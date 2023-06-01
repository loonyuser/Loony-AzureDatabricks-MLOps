# Databricks notebook source
dbutils.widgets.dropdown('Model Stage', 'None', ['Production', 'Staging','None'])

# COMMAND ----------

import mlflow

registry_uri = f'databricks://modelregistery:modelregistery'
mlflow.set_registry_uri(registry_uri)

model_name = 'churn-prediction-model'
run_id = '8ea6ddd0c60849a4a744d1e8a9f61adf'
# The default path where the MLflow autologging function stores the model
artifact_path = 'model'
model_uri = 'runs:/{run_id}/{artifact_path}'.format(run_id = run_id, artifact_path = artifact_path)

model_production_uri = 'models:/{model_name}/{model_stage}'.format(model_name = model_name, model_stage = dbutils.widgets.get('Model Stage'))

def plot(model_name, model_stage, model_version, predictions, actual_labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(actual_labels, predictions)

    display_labels = "Predicted by '%s' in stage '%s' (Version %d)" % (model_name, model_stage, model_version)
    print(display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot();

def predict_churn(model_name, model_stage):
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from mlflow.tracking.client import MlflowClient
  client = MlflowClient()
  model_version = client.get_latest_versions(model_name, stages = [dbutils.widgets.get('Model Stage')])[0].version
  model_uri = "models:/{model_name}/{dbutils.widgets.get("Model Stage")}".format(model_name = model_name, model_stage = model_stage)
  model = mlflow.pyfunc.load_model(model_uri)
  bank_cust_data = pd.read_csv('Bank_Customer.csv')
  X = bank_cust_data.drop(columns = ['churn', 'customer_id'])
  y = bank_cust_data['churn']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 123)
  sample_data = X_train[:10]
  actual_labels = y_train[:10]  
  churn_predictions = pd.DataFrame(model.predict(sample_data))
  print(churn_predictions)
  plot(model_name, model_stage, model_version, churn_predictions, actual_labels)

# COMMAND ----------

predict_churn(model_name, dbutils.widgets.get('Model Stage'))

# COMMAND ----------


