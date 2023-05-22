# RTA-Severity-Prediction
This data set is collected from Addis Ababa Sub city police departments for Masters research work. The data set has been prepared from manual records of road traffic accident of the year 2017-20. All the sensitive information have been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms algorithms.<br>
Source Link: [click here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)
## Notebook Building
The notebook is builded using Jupyter notebook the in following steps
* Importing Libraies
* Importing Dataset
* Exploratary Data Analysis (EDA)
* Hypothesis Testing
* Preprocessing
* Model Building
* Hyperparameter Tuning
* Explanaible AI (XAI)
* Select Feature for API, Again Train Model on Selected Feature
* Deploy on Web
## Importing Libraies
At the beginning of the notebook, I initiate by importing fundamental libraries such as numpy, pandas, matplotlib, and seaborn. Throughout the notebook creation process, I incorporate any necessary additional libraries. Eventually, I compile a comprehensive list of all the imported libraries, which is positioned at the beginning of the notebook.
## Importing Dataset
I retrieve the dataset from a specified source link in CSV format and bring it into the notebook as a pandas DataFrame.
## Exploratary Data Analysis (EDA)
The dataset contains 12316 records and 32 features 'Time', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation', 'Driving_experience', 'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle', 'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions', 'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Number_of_vehicles_involved', 'Number_of_casualties', 'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity', 'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement', 'Cause_of_accident' and 'Accident_severity' where 'Accident_severity' is the target feature.
Afterwards, I proceed to rename the columns to lowercase, enabling easier access throughout the entire process. I then convert the time column, which represents the accident timing, to the pandas date and time format. From there, I extract only the hour information. Following this, I generate graphs and derive the following insights.
Most of the accidents:
* involved 2 vehicles and 2 casualties
* occured on Fridays and after noon hours

Most of the drivers:
* are male and in 18-30 yrs age group
* have only went upto Junior high school and are employees
* have 5-10 yrs of driving experience

Most of the accidents happened with personally owned passenger vehicles.

Most of the drivers have met with accident on:
* two-way lanes
* tangent road with flat terrains
* Y shaped junctions
* asphalt roads
* day time
* normal weather conditions

Most of the casualties:
* happened to physically fit male drivers
* are of severity 3

The conditions on which most of the drivers met with the accident are:
* vehicle to vehicle collision
* straight movement
* no pedestrian accidents

Not keeping enough distance between the vehicles was the major cause for most of the accidents and majority of the accidents resulted in slight injury.

* Most of the accidents with fatal injuries happened between 2pm to 7pm.
* Most of the accidents with fatal injuries happened on weekends.
* Highest number of non-fatal injuries happened at 5pm.
* Highest number of non-fatal injuries happened on fridays.
* Most accidents are caused by drivers aged 18-30 and the least by drivers aged under 18.
* Proportion of fatal accidents are lower for female drivers.
* Drivers with 2-5yrs of experience caused most accidents with fatal injury and those with 5-10yrs experience caused most accidents with non-fatal injuries.
* Proportion of fatal injuries caused by vehicles with more than 10yrs of service is lower compared to non-fatal injuries.
* Though most of the accident happened around offices, a higher proportion of accidents happened around residential areas have led to fatal injuries.
* Most accidents with fatal injuries occured on undivided two-ways.
* Most accidents with non-fatal injuries occured on two-way divided with broken lines road marking
* Double carriageway has a lower proportion of fatal accidents compared to non-fatal accidents.
* Severity of accidents increases at places with no junctions and the most number of fatal injuries occured at places with no junctions.
* Crossings and Y-shaped junctions shows a decreasing trend with respect to severity of accidents though the numbers are high.
* Most of the accidents with fatal injury have happened at night.
* Collisions with pedestrians have resulted in more fatal injuries compared to non-fatal injuries.
* Accidents with 4 casualties have a huge proportion on fatal injuries compared to non-fatal injuries.
* Accidents involving drivers with 18-30yrs of age have an increasing pattern with respect to severity of injuries.
* Moving backward led to most of the accidents with fatal injuries whereas failing to keep enough distance between vehicles led to more number of accidents with non-fatal injuries.

## Hypothesis Testing
I perfome some hypothesis testing where I conclude that
* Although, the percentage of Accidents done in this sample by males is over 92% but, it doesn't actually indicate that males are more dangerous. If we calculated the probability for each gender we can deduce that both are quite the same.
* Despite the fact that speeding causes accidents. After analysis it’s found that speeding is not one of the main factors.
* There is no indication of more accidents happening on weekends
* Although it does that in daylight there are more number of accidents but the dangerous injuries percentage are almost same either be it day or night.
* Without any analysis we can clearly see that rainy weather causes more accident and also all the fatal injuries have occured under non-normal conditions are there in rainy weather
## Preprocessing
I manually encode the 'service_year_of_vehicle', 'driving_experience', 'age_band_of_driver', and 'accident_severity' features with their respective rankings. Following that, I apply Frequency Encoding to the columns with more than 10 unique values, while the remaining categorical columns are encoded using Ordinal Encoding.<br>
By creating a heatmap to visualize the correlation among the columns, I observed that 'casualty_class', 'sex_of_casualty', 'age_band_of_casualty' and 'casualty_severity' are correlated. To address this issue, I performed Principal Component Analysis with a single component PCA(n_components=1).<br>
To address the high imbalance in the target variable, I employ SMOTETomek, a technique that combines the advantages of both oversampling and undersampling. In the first step, SMOTE oversamples the minority class to augment its representation. Then, Tomek Links are utilized to eliminate potentially ambiguous or noisy samples from both the minority and majority classes. As a result, the resulting dataset achieves a more balanced distribution between the classes and reduced noise, potentially enhancing the classification performance.
## Model Selection
The data is split into a training set and a test set. Subsequently, I consider models from different categories of machine learning algorithms, including linear models such as Lasso and MLR (Multiple Linear Regression), as well as tree-based ensemble models such as RandomForest, ExtraTrees, Adaboost, and GradientBoost.
ExtraTreesClassifier provide best result hence it is selected for hyperparameter tuning.
## Hyperparameter Tuning
To fine-tune hyperparameters, I employ Optuna, an optimization tool that leverages Bayesian optimization. This approach efficiently identifies optimal hyperparameters for machine learning models by leveraging insights gained from previous evaluations. Optuna dynamically explores the hyperparameter search space, gradually converging towards the best possible combination. Users define the objective function to optimize, and Optuna suggests new configurations based on past performance. It is compatible with diverse machine learning frameworks, offers features like pruning and parallel execution, and provides visualization tools for analysis. By automating the hyperparameter optimization process, Optuna significantly reduces time and effort while enhancing model performance and overall results.
## Explainable AI (XAI)
What is XAI?
<br>
Explainable Artificial Intelligence (XAI) focuses on creating AI systems and algorithms that can provide understandable explanations for their decisions and predictions. It aims to address the lack of transparency in traditional AI models, particularly those based on machine learning or deep learning. XAI enables users to comprehend the reasoning and factors behind AI system outputs, enhancing trust, accountability, and transparency. It finds applications in domains like healthcare, finance, and autonomous vehicles. XAI techniques include post-hoc explanations, interpretable models, and interpretability methods that facilitate understanding and validation of AI systems.<br>
I utilized SHAP, a framework that explains machine learning predictions by assigning Shapley values to features. It provides a robust method to attribute feature importance, understand individual feature impact, identify important features, detect interactions, and analyze model behavior. The SHAP library offers a user-friendly implementation with visualization tools for effortless computation and interpretation.
## Select Feature for API, Again Train Model on Selected Feature
By visualizing the SHAP bar plot, I identified the top 12 features or the dependent feature. I trained these features using the ExtraTreesClassifier model and fine-tuned it. The resulting model was saved for deployment. Please note that this process was performed in another notebook and will be updated soon.
## Deploy on Web
I utilize Streamlit for building the API, and the application is deployed on Amazon EC2.
