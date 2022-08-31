# CloudSkillsChallengeStudySession
#  Use Automated Machine Learning in Azure Machine

## Module Source

[Use Automated Machine Learning in Azure Machine](https://docs.microsoft.com/en-us/learn/modules/use-automated-machine-learning/)

## Goals

In this workshop, we will learn how to train **regression models** - machine Learning models that can predict numerical values. In this workshop, we will focus on predicting **car prices**, but you can optionally try to use the same approach to predict **number of likes in instagram**.

| **Project Goal**              | Learn how to predict car prices / instagram likes based on data |
| ----------------------------- | --------------------------------------------------------------------- |
| **What will you learn**       |  Train a classification model with no-code AutoML in the Azure Machine Learning studio |
| **What you'll need**          | [Azure Subscription](https://azure-for-academics.github.io/getting-azure/) |
| **Duration**                  | *1 hour*                                                                |
| **Slides** | [Powerpoint](https://), [SpeakerDeck](https://) |
| **Author** | [AYUSH JHA](https://ayush-jha.me/) |


## Video walk-through

*coming soon*

## What students will learn

In this project, you will learn how to train a classification model with no-code AutoML using Azure Machine Learning automated ML in the Azure Machine Learning studio. This classification model predicts if a client will subscribe to a fixed term deposit with a financial institution.

## Prerequisites

For this workshop, you need to have an [Azure Account](https://azure-for-academics.github.io/getting-azure/). You may have one from your university, otherwise get [Azure for Students](https://azure.microsoft.com/free/students/?WT.mc_id=academic-49822-dmitryso), [GitHub Student Developer Pack](https://education.github.com/pack) or an [Azure Free Trial](https://azure.microsoft.com/free/?WT.mc_id=academic-49822-dmitryso).

> Learn more about creating an Azure Account at [Microsoft Learn](https://docs.microsoft.com/learn/modules/create-an-azure-account/?WT.mc_id=academic-49822-dmitryso)

## Running the Workshop

The workshop closely follows [Create a Regression Model with Azure Machine Learning designer](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-56424-dmitryso) Microsoft Learn module. You may want to open the module in a separate tab/window and follow along with the instructions below.

## Milestone 1: Create Machine Learning Workspace and Compute Resources

To work with Azure Machine Learning, you need a **Workspace**. Follow [the instructions on Learn](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/2-create-workspace/?WT.mc_id=academic-56424-dmitryso) to create the workspace interactively using Azure Portal.

> If you are a developer and prefer doing things via code, you can also create a workspace [through command-line](https://docs.microsoft.com/azure/machine-learning/how-to-manage-workspace-cli/?WT.mc_id=academic-56424-dmitryso) or [through Python SDK](https://docs.microsoft.com/azure/machine-learning/how-to-manage-workspace?tabs=python&WT.mc_id=academic-56424-dmitryso)

After you have create the workspace, open it in [Azure Machine Learning Studio](https://ml.azure.com/?WT.mc_id=academic-56424-dmitryso)

Then, you need to create **compute resources**. Follow [the instructions on Learn](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/3-create-compute/?WT.mc_id=academic-56424-dmitryso) to create two types of compute resources:

* **Compute instance** is a virtual machine that you can use to run Jupyter notebooks, a tool that many data scientists use. We will use it in order to test our model towards the end of the module.
* **Compute cluster** is a scalable compute that automatically allocated to train your Azure Machine Learning models. It will be the main compute that we will use for training.

> **Note**: With newer release of Azure Machine Learning studio, you can avoid creating compute cluster altogether, and use compute instance to run your experiment. Using clusters to run experiments makes sense when you need fast compute that you want to schedule automatically. In our case, you can save time by not using compute cluster.

## Milestone 2: Run an automated machine learning experiment


## Milestone 3: Train the model and evaluate results

Once you have the data, you can train machine learning model to predict prices. Follow [the instructions on Learn](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/5-create-training-pipeline/?WT.mc_id=academic-56424-dmitryso) to split the data into train and test dataset, and train the model on the training data.

You can then use **Score model** block to apply the model to the test dataset. Finally, follow [next learn unit](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/5-create-training-pipeline/?WT.mc_id=academic-56424-dmitryso)
to add **Evaluate model** block to calculate some statistics.

Your final experiment should look like this:

![Final Experiment](images/train-experiment.png)

As a result, we should get **coefficient of determination** around 93%. This coefficient shows how much predicted prices are related to the real ones in the test dataset, i.e. how accurately are we able to predict. In absolute values, **mean absolute error** shows the average difference between expected and predicted price, which is around $2k in our case. It is not too bad, given the average price is $13k, so we are making around 15% prediction error.

## Milestone 4: Build and Deploy Inference Pipeline

Once we have trained the model, we need to be able to apply it to our own data to make predictions. This can be done using **Inference pipeline**, which we will build next. Follow [next learn unit](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/7-inference-pipeline/?WT.mc_id=academic-56424-dmitryso) to build the pipeline.

Pipeline is constructed as a separate experiment, which includes two special blocks: **web service input** and **web service output**. They represent input and output data of the model, and there can be any transformations in between. In our case, we need to perform the same transformation steps (including normalization) on input data, and then call the model.

We also use **Python code block** in order to simplify output data from the model. This is not strictly necessary, but it shows how we can embed custom code into the pipeline to perform any data processing we need.

You can now **Run** the pipeline to make sure it does predictions correctly. It will give you predicted price for the sample data that we have provided in the beginning of the pipeline.

![Inference Pipeline](images/inference-pipeline.png)

Once we have run the pipeline, we can **deploy it**, so that the model can be called through the internet as a web service. Follow [next learn unit](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/8-deploy-service/?WT.mc_id=academic-56424-dmitryso) to deploy the service into **Azure Container Instance**.

Azure ML offers two ways to deploy the model:

* **Azure Container Instance** (ACI) means that there will be one container on a virtual machine that serves your model. It is not scalable, so we can recommend it for testing, but not for production environments.
* **Azure Kubernetes Service** (AKS) deploys the model onto Kubernetes cluster (which you can create in the **Compute** section of the portal), which makes the model scale under load. 

In our workshop, we will deploy to ACI.

## Milestone 5: Test the Model

Once the model has been deployed, we can call it as REST service through the internet from anywhere, eg. from mobile application, web application, etc. In our case, we will use a piece of Python code to test our web service. Follow [this learn unit](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/8-deploy-service/?WT.mc_id=academic-56424-dmitryso).

To run Python code, we will use **Notebooks** feature of Azure Machine Learning. It allows you to create and run **Jupyter Notebooks** - documents that include text and executable Python code. In the instructions, Python code for calling the service is provided, you need to create the notebook and copy-paste the code there. You also need to provide correct **service endpoint URL** and **key**.

Once you run the code, it will call the REST service, and you should see the predicted price as a result. In the similar manner, you can call the same service from any other programming language, making the price prediction functionality available in the wide range of applications.

## Milestone 6 [Optional]: Try to train different model

To understand the process even better, it is always a good idea to repeat the same exercise on different data, in different problem domain. For example, you can try to predict the number of likes of an instagram photo. You can use [Instagram Likes Dataset](https://github.com/shwars/NeuroWorkshopData/tree/master/Data/instagram).

* [instagram.csv](https://github.com/shwars/NeuroWorkshopData/blob/master/Data/instagram/instagram.csv) contains the result of processing of Instagram posts of a famous blogger. Each photograph was processed using [Computer Vision Cognitive Service](https://azure.microsoft.com/services/cognitive-services/computer-vision/?WT.mc_id=academic-56424-dmitryso) to extract main category, tags, background/foreground color and some other properties.

likes|main_category|no_faces|avg_age|date|adult_score|color_fg|color_bg|tags|objects
-----|-------------|--------|-------|----|-----------|--------|--------|----|-------
321112|people|0|0|2017-06-24|0.005|Pink|Pink|woman person dress clothing|person
245154|building|0|0|2017-06-25|0.003|White|Grey|sky outdoor|plant
279328|others|1|53.0|2017-07-03|0.004|White|Black|text|person
299499|people|0|0|2017-07-05|0.069|Black|Pink|person food outdoor

* [instagram-onehot.csv](https://github.com/shwars/NeuroWorkshopData/blob/master/Data/instagram/instagram_onehot.csv) is the same dataset, but most frequently used tags are represented by **one-hot encoding**, i.e. they have 0/1 in the corresponding column depending on whether corresponding tag is associated with the image or not.


Before you start building an experiment, you need to upload the dataset to Azure ML portal. Go to **Datasets** -> **Create Dataset** -> **From Web Files**. In the following dialog, provide one of the following URLs to the dataset file:
* instagram.csv - `https://raw.githubusercontent.com/shwars/NeuroWorkshopData/master/Data/instagram/instagram.csv`
* instagram_onehot.csv - `https://raw.githubusercontent.com/shwars/NeuroWorkshopData/master/Data/instagram/instagram_onehot.csv`  

![Create dataset from web](images/dataset-from-web.png)

After you create the dataset, go to **Designer** and construct the experiment similar to the one for car prices prediction:
* Start with the uploaded dataset
* Use **Select columns from Dataset** to ignore some of the columns. You may want to ignore textual columns such as **tags** or **objects**.
* Since the data includes a lot of *categorical* features that are likely to have non-linear effect on the result, use decision trees family of models, for example, **Decision Forest Regression**.

Train the model and see if you can get reasonably accurate predictions.

You can also explore some other datasets from [Kaggle](http://kaggle.com/datasets), for example:
* A [dataset of Possums](https://www.kaggle.com/datasets/abrambeyer/openintro-possum) physical characteristics. You can try predicting the age of a possum based on head length, skull width, total length, etc.
* A dataset of [medical insurance costs](https://www.kaggle.com/datasets/mirichoi0218/insurance), where you can try to predict insurance costs based on different parameters.
* Predicting [real estate prices](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction)

## Milestone 7: Clean Up

Once you have done the experiments, it is important to get rid of the resources that you are not using in order not to spend credits from your Azure subscription. Make sure you do the following:

* Delete the ACI instance of your predictive service.
* Stop the Compute Instance. If you are planning to use Azure ML often, it makes sense to set up the schedule to automatically shut down the compute instance each day at the end of the day, so that you do not leave it running and unused for long periods of time.
* If you are not planning any experiments with Azure ML in the nearest future - delete Azure ML workspace altogether. Even when no compute is running, existing cloud resources still incur some storage costs.

## Next steps

Now that you know how to use Azure ML Designer, you can explore other components of Azure ML:

* [Automated Machine Learning](https://docs.microsoft.com/learn/modules/use-automated-machine-learning/?WT.mc_id=academic-56424-dmitryso) is another Low Code / No Code approach that automatically tries to find the best model by trying out different options and hyperparameters
* Learn how to [Create Classification Models](https://docs.microsoft.com/learn/modules/create-classification-model-azure-machine-learning-designer/?WT.mc_id=academic-56424-dmitryso) in Azure ML designer
* Learn how you can [Explain machine learning models](https://docs.microsoft.com/learn/modules/explain-machine-learning-models-with-azure-machine-learning/?WT.mc_id=academic-56424-dmitryso) to understand which features play more important role in the result.


## Practice

You have learnt how to train a model and make it usable through REST API. You can build a project that will expose your trained regression model to the users by building a mobile or web application. Team up with some peers with programming experience to create an end-to-end predictive application:

* Select a problem domain that interests you. Get an inspiration from datasets available on Kaggle or elsewhere
* Train and deploy the model the way you have done in this workshop
* Build a mobile or web application that consumes the API. You can further learn [how to consume REST API from Xamarin](https://docs.microsoft.com/learn/modules/consume-rest-services/?WT.mc_id=academic-56424-dmitryso).


## Followup

* [How to Learn Data Science without Coding](https://soshnikov.com/azure/how-to-learn-data-science-without-coding/) - blog post describing another No Code/Low Code approach to training machine learning models - [Automated ML](https://azure.microsoft.com/services/machine-learning/automatedml/?WT.mc_id=academic-56424-dmitryso)

## Feedback

Be sure to give [feedback about this workshop](https://forms.office.com/r/MdhJWMZthR)!

[Code of Conduct](../../CODE_OF_CONDUCT.md)
