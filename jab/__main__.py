import os
import argparse
import json
import logging
from getpass import getpass
from .modules.naukri import NaukriBot
from .modules.model import ChatbotBuild

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("jobautobot")

def main():
    parser = argparse.ArgumentParser(description="Apply using NaukriBot")
    parser.add_argument("--apply", action="store_true", help="To start applying")
    parser.add_argument("--train", action="store_true", help="To train the model")
    parser.add_argument("--email", required=True, help="email address associated with naukri account")
    parser.add_argument("--filters", action="store_true", help="boolean weather to apply filters or not")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    email = args.email
    username = args.email
    logger.info(f"Starting jobautobot with email: {email}")
    
    if args.apply:
        filt = args.filters
        logger.info(f"Apply mode with filters: {filt}")
        #password = getpass("Password: ")
        password = "Haryanao@123"
        #number = int(input("Number of jobs to apply (default=10): ") or 10)
        number = 30
        logger.info(f"Will apply to {number} jobs")
        
        # Initialize NaukriBot with debug flag
        nb = NaukriBot(email, password, username, number, debug=args.debug)
        
        if filt:
            logger.info("Starting filter-based job application")
            print("Choose your filters to apply::::")
            search = input("Job search criteria or keyword to search for (required if applying filters): ")
            experience = input("Years of experience (optional, leave blank if none): ")
            location = input("Job location (optional, leave blank if none): ")
            jobAge = input("Age of job posting in days (default set to 3 days): ")
            experience = int(experience) if experience else None
            jobAge = int(jobAge) if jobAge else None
            logger.debug(f"Applying with filters - search: {search}, exp: {experience}, location: {location}, jobAge: {jobAge}")
            nb.filter_apply(search, experience, location, jobAge)
        else:
            logger.info("Starting tab-based job application")
            # tab = input(f"Choose from these options to start: {nb.tabs} : ")
            tab = "apply"
            logger.debug(f"Selected tab: {tab}")
            nb.start_apply(tab)
    elif args.train:
        logger.info("Starting model training")
        chb = ChatbotBuild(email)
        chb.train_model()
        training_data = chb.training_data
        output_path = f"./jab/data/{username}/training_data.json"
        logger.info(f"Saving training data to {output_path}")
        with open(output_path, "w+") as f:
            f.write(json.dumps(training_data, indent=4))
        logger.info("Training completed successfully")
    else:
        print("Use --apply to start applying or --train to train the model")

if __name__ == "__main__":
    main()