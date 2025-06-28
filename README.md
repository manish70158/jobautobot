# jobautobot

jobautobot is designed to help users find and apply jobs on Naukri platform by automating the whole process.
It uses playwright for web automation tasks, and to answer questions while applying for jobs, it uses a Neural network model from tensorflow for classification trained upon user data.

# Getting Started

### Prerequisites
- Python
- pip
- venv

### Installation

step 1: Clone the repository
```bash
    git clone https://github.com/aman-dayal/jobautobot.git
```
step 2: Navigate to the folder containing the code  
```bash
cd jobautobot
```
step 3: create a virtual environment and activate it

For linux
```bash
python3 -m venv jbvnv && source jbvnv/bin/activate
```
For Windows powershell
```bash
python -m venv jbvnv &&  jbvnv/scripts/activate.ps1
```
For Windows cmd
```bash
python -m venv jbvnv
call jbvnv/scripts/activate
```
step 4: Install the required packages
```bash
pip install -r requirements.txt
```
step 5: Install the required playwright packages and browser binaries
```bash
playwright install
```
or
```bash
playwright install --with-deps
```
or
```bash
playwright install chromium
```
### Usage

Now that setup is complete next step is to train the model that will be used to answer questions while applying for jobs. Navigate to the folder jab/data/user_data.json and fillout the json with your data in the values corresponding to the keys.
Keep the dob in the format DD/MM/YYYY .

Finally its time to train the model using the data you just updated.

Make sure you are in the root directory of the project and the virtual environment is activated.

To train the model run:
```bash
# python -m jab --email a.manish1689@gmail.com --train
python -m jab --email kumar.manish1689@gmail.com --train
```
Replace your-email@gmail.com with your actual email, this email is used to identify your trained model and data when sending out job applications. The model will be trained and saved in the jab/data/your-email@gmail.com models directory.

You are all set and ready to send out your first application using jobautobot or jab. To send out a job application run:
```bash
# python -m jab --email a.manish1689@gmail.com --apply
python -m jab --email kumar.manish1689@gmail.com --apply
```
If you would like to apply filters to search for the jobs run:
```bash
python -m jab --email your-email@gmail.com --apply --filters
```
follow along the prompts to put in your password and select the filters you would like to apply.

Complete documentation can be found here: https://aman-dayal.github.io/Documentation-for-JobAutobot
