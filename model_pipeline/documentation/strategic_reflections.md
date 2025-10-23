# Strategic Reflections

---

### 1. Model Failure: Your model underestimates delivery time on rainy days. Do you fix the model, the data or the business expectations?

>There is nothing more dangerous than assuming that the data is correct and start fixing the model without fact-checking that the data is, in fact, accurate. There is no model nor even thinking on performance without fixing the input data.
>If, after validating that the data is good the model still underperforms on rainy days, meaning that it no longer fits the problem and is starting to drift, it is part of the data science cycle to start over experimenting and evaluating how it can be improved.

### 2. Transferability: The model performs well in Mumbai. It's now being deployed in Sao Paulo. How do you ensure generalization?
>It is mandatory to do an EDA for both cities and see if they have similar conditions (or not) first, also evaluating model performance on test data and reaching an agreement with managers from the region backed up by data (results are not leaving any doubt) before moving into deploying a model into production, which represents costs associated to running it on cloud services, integrations required and time spent by the teams related to this.

### 3. GenAI Disclosure: Generative AI tools are a great resource that can facilitate development, what parts of this project did you use GenAI tools for? How did you validate or modify their output?
>GenAI tools are helpful if they are in the right hands who can spot when content is correct or not. I use Copilot regularly in my job (only approved GenAI tool) to get some templates that can help me as a start point to continue with other more sophisticated tasks very related to my field and expertise that only people with experience and domain knowledge can deliver at this time (October 2025). 
>For this case study, I used GenAI tools (OpenAI, Gemini and the IDE that I used, Pycharm, that has a not so bad autocomplete function) mainly to avoid starting from scratch as it could represent an important amount of time for tasks where the tools have reached a good level already. e.g. quick templates for plotting charts using seaborn and matplotlib, debugging some functions, draft of some scripts, where you can modify later and change or improve according to your needs.

### 4. Your Signature Insight: What's one non-obvious insight or decision you're proud of from this project? 
>Thinking on reproducibility, auditing, governance and future proofing, I've used toml file as the main repository for constants used in the project. Also having most of the functions in one single file Global.py, for the same reasons, reusability of functions. About the findings, the EDA part regarding the IQR analysis was very helpful for me to understand the challenges present in the data for this problem that were later confirmed when validating the model performance results.

### 5. Going to Production: How would you deploy your model to production? What other components would you need to include/develop in your codebase? Please be detailed on each step of the proces, and feel free to showcase some of these components in the codebase.
>There is still a lot of work to do in terms of finetuning the model and experimenting to reduce the error. 
>
>Having mentioned this, it will be required to have a git (e.g. bitbucket) as repository for the source code to track changes and versioning. 
>
>Also, I see the need of creating a pipeline orchestration that aligns with the cloud service where the model is planned to be deployed (e.g. AWS) where automated retraining when required is possible. 
>
> Some security checks are required e.g. AWS secrets manager, to be able to connect to the service.
>
>Perhaps if it is possible to enrich the data and there are large volumes of information, create preprocessing functions based on pyspark for parallel processing and launching this job on the cloud on an EMR cluster, could work for this purpose.
>
>Migrating the project and working directly on the cloud for experimentation in Sagemaker could be a good option to speed up this phase, adding model registry to keep the inventory of that, or keep using the aws SDK to connect between the local environment and the cloud. 
>
> About configuring the endpoint to make the model available for real-time predictions, activating alerts based on up-time and model monitoring.
>
>To facilitate the deployment, project code could be packaged using Docker and then add it to platform. 
