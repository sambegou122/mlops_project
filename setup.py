from setuptools import setup, find_packages

setup(
    name='sentiment analyzer',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={"":"src"},    
    install_requires=[
        # project dependancies
    ],
    entry_points={
        'console_scripts': [
            'predict=sentiment_analyzer.predict:predict',
            'promote=sentiment_analyzer.promote:promote',
            'retrain=sentiment_analyzer.retrain:retrain',
            'app=sentiment_analyzer.webapp.app:app',
            'get_mlflow_model=sentiment_analyzer.webapp.get_mlflow_model:main',
            'hf_export=sentiment_analyzer.hf_export:hf_export',
        ],
    },
)