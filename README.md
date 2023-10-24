---
contributors:
  - Edwige Elysee
  - Theodore Quansah
---

# git_lingua

This project aims to predict the main programming language of a repository, using only the text of the README.me file.

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





