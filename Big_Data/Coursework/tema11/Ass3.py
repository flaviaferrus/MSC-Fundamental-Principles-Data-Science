from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def print_name( params,**context):
    print(context)
    name= context['task_instance'].xcom_pull(task_ids='Py_task_3')
    print(name,params["Rosado_Ferrus"])
    

