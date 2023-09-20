# fake-jobs

## Kaggle competition of the 'Machine Learning' course (UB Data Science MSc, 2022)

We conform the team "greMLins" and the competition's title is "Predicting fake job ads". More details can be found [here](https://www.kaggle.com/competitions/inclassfakejobs/overview/).

### Description

Are you able to spot fake ads?

In this dataset we have ads textual descriptions as well as contextual and metadata information. The goal is to identify which of them are fake ads.

#### Features

- ``job_id``: ID of the Ads
- ``title``: title of the ad
- ``location``: Name of the location: 
- ``department``: Name of the department in the company where the candidate will be hired.
- ``salary_range``: Range of salary…
- ``company_profile``: Description of the company
- ``description``: Description of the job
- ``requirements``: list of mandatory requirements for application
- ``benefits``: additional benefits to the job description
- ``telecommuting``: True if telework is available
- ``has_company_logo``: True if the ad shows the logo
- ``has_questions``: True if screening questions are present
- ``employment_type``: categorical description of the required dedication of the offering (full-time, part-time, …)
- ``required_experience``: categorical with required entry level experience title
- ``required_education``: categorical with required education required
- ``industry``: categorical with type of industry (telecom, automotive, …)
- ``function``: categorical summarizing job function (sales, it, consulting, engineering, …)
- ``requireddoughnutscomsumption``: normalized average amount of doughnuts that the employee is expected to consume every day.

#### Labels
- ``fraudulent``: corresponds to the desired feature to be predicted. (0: non-fraudulent, 1: fraudulent)

### Evaluation

The evaluation metric for this competition is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision and recall.

The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

#### Submission Format
For every ad in the dataset, submission files should contain two columns: Id and Category. Id corresponds to the id of the data sample (not the ad id). And Category is an integer with value 0 or 1 according to the prediction.

The file should contain a header and have the following format:

```console
Id,Category
1,1
```

#### Acknowledgements
The University of the Aegean | Laboratory of Information & Communication Systems Security

## Run the scripts

### Requisites

A dedicated conda environment should be created to be able to run the scripts:

```console
$ conda create --name fake-jobs python=3.9
$ conda activate fake-jobs 
$ conda install numpy pandas matplotlib colorama -y
$ conda install -c conda-forge xgboost lightgbm -y
$ conda install -c anaconda nltk
$ conda install -c conda-forge spacy
$ python -m spacy download en_core_web_sm
```

..
    $ pip install -U spacy
    $ python -m spacy download en_core_sci_lg

### Execute

Once with the environment activated, place yourself in the package folder and just type the following to run the scripts:
```console
python main.py
```

## Documentation

### Requisites

To the former (activated) environment we shall add the following packages for the documentation be compiled:

```console
$ conda install -c anaconda sphinx numpydoc sphinx_rtd_theme recommonmark python-graphviz 
$ pip install --upgrade myst-parser
```

### Re-compile the build

To re-build the .html documentation we will run:
 ```console
$ cd docs
$ make html
```


