import dagshub
dagshub.init(repo_owner='sfgrahman', repo_name='mlops_complete_project', mlflow=True)

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/sfgrahman/mlops_complete_project.mlflow")
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)