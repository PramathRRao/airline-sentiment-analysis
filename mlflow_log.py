import mlflow

mlflow.set_experiment("airline-sentiment-roberta")

with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model_name", "cardiffnlp/twitter-roberta-base-sentiment")
    mlflow.log_param("num_labels", 3)
    mlflow.log_param("max_length", 128)
    
    # Log evaluation metrics from Week 2
    mlflow.log_metric("f1_negative", 0.90)
    mlflow.log_metric("f1_positive", 0.78)
    mlflow.log_metric("f1_neutral", 0.64)
    mlflow.log_metric("accuracy", 0.85)