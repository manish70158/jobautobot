import os
import argparse
import json
from getpass import getpass
from .modules.naukri import NaukriBot
from .modules.model import ChatbotBuild

def main():
    parser = argparse.ArgumentParser(description="Apply using NaukriBot")
    parser.add_argument("--apply",action="store_true",help="To start applying")
    parser.add_argument("--train",action="store_true",help="To train the model")
    parser.add_argument("--email",required=True,help="email address associated with naukri account")
    parser.add_argument("--filters", action="store_true", help="boolean weather to apply filters or not")
    args = parser.parse_args()
    email = args.email
    username = args.email
    if args.apply:
        filt = args.filters
        print("filters",filt)
        password = getpass("Password: ")
        number = int(input("Number of jobs to apply (default=10): ") or 10)
        nb = NaukriBot(email,password,username,number)
        if filt:
            print("Choose your filters to apply::::")
            search = input("Job search criteria or keyword to search for (required if applying filters): ")
            experience = input("Years of experience (optional, leave blank if none): ")
            location = input("Job location (optional, leave blank if none): ")
            jobAge = input("Age of job posting in days (default set to 3 days): ")
            experience = int(experience) if experience else None
            jobAge = int(jobAge) if jobAge else None
            nb.filter_apply(search,experience,location,jobAge)
        else:
            tab = input(f"Choose from these options to start: {nb.tabs} : ")
            nb.start_apply(tab)
    elif args.train:
        chb = ChatbotBuild(email)
        chb.train_model()
        training_data = chb.training_data
        with open(f"./jab/data/{username}/training_data.json","w+") as f:
            f.write(json.dumps(training_data,indent=4))
    else:
        print("Use --apply to start applying or --train to train the model")

if __name__ == "__main__":
    main()