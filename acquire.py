"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
import base64
import pandas as pd
import requests

from env import github_token, github_username
from typing import Dict, List, Optional, Union, cast



def search_github_repositories(search_query, repository_type="repositories", per_page=100):
    # Define the base URL for the GitHub API
    base_url = "https://api.github.com/search/repositories"

    # Set up headers with your GitHub token and user-agent
    headers = {
        "Authorization": f"token {github_token}",
        "User-Agent": github_username
    }

    # Initialize an empty list to store repositories
    all_repositories = []

    # Initialize variables for pagination
    page = 1
    total_repos = 0

    while total_repos < 1000: 
      
        params = {
            "q": search_query,
            "type": repository_type,
            "per_page": per_page,
            "page": page
        }

        # Send a GET request to the GitHub API
        response = requests.get(base_url, headers=headers, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            # Extract the repositories from the current page
            repositories = data.get("items", [])
            # Append the current page's repositories to the list
            all_repositories.extend(repositories)
            # Update the total count of repositories retrieved
            total_repos += len(repositories)
            # Increment the page number for the next page
            page += 1
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break

    return all_repositories




def get_repo(search_query, per_page=100):
    # Check if the CSV file exists
    if os.path.exists('github_repos.csv'):
        # If the CSV file exists, read the data from the file
        df = pd.read_csv('github_repos.csv')
    else:
        # Use the existing function to fetch GitHub repositories
        repositories = search_github_repositories(search_query, "repositories", per_page)
        
        repo_data = []
        
        for repo in repositories:
            repo_info = {
                "Name": repo["name"],
                "URL": repo["html_url"],
                "Description": repo["description"],
                "Readme": "",
            }
        
            if repo["has_wiki"]:
                readme_url = f"https://api.github.com/repos/{repo['full_name']}/readme"
                headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
                response = requests.get(readme_url, headers=headers)
        
                if response.status_code == 200:
                    readme_data = response.json()
                    encoded_readme = readme_data.get("content", "")
                    decoded_readme = base64.b64decode(encoded_readme).decode('utf-8')
                    repo_info["Readme"] = decoded_readme
        
            repo_data.append(repo_info)
        
        # Create a DataFrame from the repo_data
        df = pd.DataFrame(repo_data)
        
        # Save the DataFrame to a CSV file for future use
        df.to_csv('github_repos.csv', index=False)
    
    return df




