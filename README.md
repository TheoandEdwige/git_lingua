---
contributors:
  - Edwige Elysee
  - Theodore Quansah
---

# git_lingua


## Project Goal
This project aims to predict the main programming language of a repository, using only the text of the README.me file.

With hopes to showcase mastery of the data sciecne pipline and it's tools. As well as demonstrating practical application of Natural Language Processisng best practices.


## Project plan:

1. Collect data from GitHub repositories.
2. Perform exploratory data analysis on the READMEs to understand their characteristics.
3. Build and evaluate machine learning models for programming language prediction.


## Project Deliverables
The deliverables of this project are:
- A machine learning model for text classification.
- Insights into the relationships between README text and programming languages.
- A well-documented Jupyter Notebook.
- Presentation slides summarizing the project's findings.
- A comprehensive, intuitivley navigated README.md file.
- A .csv file with predictions


## Enviornment Preperation

In order to recreate these steps and run this jupyter notebook follow the below steps.

1. Clone repository onto your drive.
2. Create an env.py file.
3. Make a github personal access token.
    a. Go here and generate a personal access token https://github.com/settings/tokens
- You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
    b. Save it in your env.py file under the variable `github_token`
    c. Add your github username to your env.py file under the variable `github_username`
    d. both variables in your env.py file should be a string
    

Your enviornment is now set up to run the project. You may need to install libraries using pip install or an enviornment manager like conda.

Link to conda:
https://docs.conda.io/en/latest/


### Data Dictionary

| Field             | Description                                                 |
|-------------------|-------------------------------------------------------------|
| **name**          | Repository name                                             |
| **language**      | Programming language used                                   |
| **readme**        | Text content of the README file                             |
| **UniqueWords**   | Count of unique words in the README                         |
| **readme_words**  | List of words in the README                                 |
| **readme_word_count** | Total word count of the README                           |
| **learning**      | Binary flag indicating if the repo is related to learning  |
| **encoded_target**| Encoded target variable        |


### Programming Languages

| Field               | Description                            |
|---------------------|----------------------------------------|
| **Python**          | Python programming language            |
| **Jupyter Notebook**| Jupyter Notebook environment           |
| **Java**            | Java programming language              |
| **Go**              | Go programming language                |
| **Common Lisp**     | Common Lisp programming language       |
| **Ruby**            | Ruby programming language              |
| **HTML**            | HyperText Markup Language              |
| **R**               | R programming language                 |
| **C++**             | C++ programming language               |
| **PHP**             | PHP programming language               |
| **C#**              | C# programming language                |
| **JavaScript**      | JavaScript programming language        |
| **WebAssembly**     | WebAssembly language                   |
| **Scheme**          | Scheme programming language            |
| **C**               | C programming language                 |
| **Objective-J**     | Objective-J programming language       |
| **V**               | V programming language                 |
| **Smalltalk**       | Smalltalk programming language         |
| **Matlab**          | MATLAB programming environment         |
| **Rust**            | Rust programming language              |
| **PureBasic**       | PureBasic programming language         |
| **TeX**             | TeX typesetting system                 |
| **CMake**           | CMake build system                     |
| **Objective-C**     | Objective-C programming language       |
| **Julia**           | Julia programming language             |
| **MATLAB**          | MATLAB programming environment         |
| **TypeScript**      | TypeScript programming language        |
| **Swift**           | Swift programming language             |
| **HLSL**            | High-Level Shading Language            |
| **Clojure**         | Clojure programming language           |
| **GDScript**        | GDScript programming language          |
| **Idris**           | Idris programming language             |
| **Vue**             | Vue.js framework                       |
| **Arduino**         | Arduino programming environment        |
| **Makefile**        | Make build automation tool             |
| **Roff**            | Roff typesetting system                |
| **Lua**             | Lua programming language               |
| **NetLogo**         | NetLogo modeling environment           |
| **CLIPS**           | CLIPS rule-based programming language  |
| **Mustache**        | Mustache templating system             |
| **Shell**           | Shell scripting language               |
| **Prolog**          | Prolog programming language            |
| **Scala**           | Scala programming language             |
| **Dart**            | Dart programming language              |
| **Crystal**         | Crystal programming language           |
| **ASP**             | Active Server Pages                    |
| **PostScript**      | PostScript page description language  |


### Project File System

| Field                 | Description                                                |
|-----------------------|------------------------------------------------------------|
| **README.md**         | Project documentation file                                 |
| **explore.py**        | Exploration script                                         |
| **final_report.ipynb**         | Final project notebook                           |
| **mvp.ipynb**         | Minimum Viable Product notebook                            |
| **__pycache__**       | Compiled Python files                                      |
| **github_repo.csv**   | CSV file containing GitHub repository data                 |
| **nlpacquire.py**     | Scripts for acquiring NLP data                              |
| **acquire.py**        | Data acquisition scripts                                 |
| **github_repos.csv**  | Another CSV file containing GitHub repository data          |
| **prepare.py**        | Data preparation script                                    |
| **bad csv query**     | Placeholder for malformed CSV queries                      |
| **modeling.py**       | Script for data modeling                                   |
| **scrapnotebook.ipynb**| Notebook for web scraping                                  |
| **edwige_scratch.ipynb**| Scratch notebook for experimental code                    |
| **mvp-Copy1.ipynb**   | Copy of Minimum Viable Product notebook                    |
| **wrangle.py**        | Data wrangling script                                      |
| **env.py**            | Environment variables and settings                         |
| **mvp-Copy2.ipynb**   | Another copy of Minimum Viable Product notebook            |
| **predictions.csv**   | Best models predictions on the test dataset            |


## Exploration Questions and Awnsers

#### 1) Does the programming language used in a GitHub repository affect the length of the README file (in terms of word count)?

We failed to reject the null hypothesis. There is no significant difference in word counts between programming languages in GitHub repositories. 

A Mann-Whitney U test yielded a z score of 5005.0 and a p-value of approximately 0.5836.

#### 2) Does the frequency of specific words in a README file have an impact on the choice of programming language for a repository?

We rejected the null hypothesis: There is an association between programming language and specific word presence.

The chi-squared test yielded a chi-statistic of approximately 12150.15 and a p-value of (2.2625e-13). 

There is an association between the programming language chosen for a repository and the presence of specific words in the README files.


#### 3) Are there specific words associated with each of our most popular programming languages?

We rejected the null hypothesis for all of our top laguages. They all had words that were more frequently used in their respective READMES's


# Overall Project Conclusion

## Project Goals and Approach

The goal of this project was to develop a predictive model that identifies the main programming language of a repository based on the README text. To achieve this goal, we followed a structured approach:

1. **Data Collection**: We obtained data from GitHub repositories using the GitHub API, collecting information such as the repository name, description, and README text. Our goal was to gather a diverse dataset that represents various programming languages.

2. **Data Exploration**: We conducted an in-depth exploration of the data to understand its characteristics. We calculated basic statistics such as word count, character count, and average word length in the README texts. Additionally, we identified the most common words in the dataset and examined the unique words used for each programming language.

## Key Findings

### Data Exploration

Our data exploration revealed several key findings:

- The dataset contained a total of 784 README texts with 783 unique texts. However, two texts were identical.
- The most common words in the README texts included "learning," "data," "machine," and others, highlighting their prevalence in the programming community.
- The analysis of unique words showed distinct patterns for different programming languages. For example, "Python" was highly associated with Python-related READMEs.

### Model Development

We trained and evaluated four machine learning models on the data:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression

The models were assessed based on accuracy, precision, recall, and F1-score on a validation dataset. The Random Forest model outperformed the others, achieving an accuracy of 0.4331.

## Recommendations

Based on our findings, we make the following recommendations:

1. **Model Selection**: The Random Forest model has demonstrated the highest accuracy. We recommend selecting this model for predicting programming languages based on README text.

2. **Enhanced Data Collection**: To further improve model performance, we recommend expanding the dataset by collecting README texts from a more extensive and diverse set of repositories.

3. **Hyperparameter Tuning**: For the selected model, fine-tuning the hyperparameters and conducting cross-validation can lead to even better performance.

4. **Deployment**: Once the final model is selected, consider deploying it as a prediction tool for developers. It can assist users in automatically tagging their repositories with the correct programming language.

## Next Steps

If we had more time and resources, we would consider the following next steps:

1. **Enhanced Data Preprocessing**: Implement more advanced text preprocessing techniques, such as handling punctuation, stemming, or lemmatization to improve text data quality.

2. **Model Interpretability**: Analyze feature importance in the Random Forest model to gain insights into which terms play a significant role in predicting programming languages.

3. **Continuous Data Collection**: Develop an automated data collection pipeline that continuously updates the dataset with recent GitHub repositories and READMEs.

4. **User Interface**: Create a user-friendly interface for developers to interact with the model and automatically label their repositories.

This project has laid the foundation for a valuable tool that can assist developers and the programming community. By implementing the recommendations and next steps, we can refine and expand this tool to further contribute to the developer community.