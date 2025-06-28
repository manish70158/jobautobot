import time
import re
import numpy as np
import nltk 
import time
import json
import logging
from urllib.parse import urlparse, urlunparse
from playwright.sync_api import sync_playwright, expect
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# Set up module logger
logger = logging.getLogger("jobautobot.naukri")

class ChatbotModel():
    def __init__(self, user_data):
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = ['?', '!', '.', ',']
        self.user_data = user_data
        self.words = []
        self.classes = []
        self.load_data()
        self.load_model()

    def load_data(self):
        for intent in self.user_data:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        self.words = sorted(set([self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]))
        self.classes = sorted(set(self.classes))
    
    def load_model(self):
        model_path = f"./jab/data/{user}/model.keras"
        self.model = tf.keras.models.load_model(model_path)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints):
        tag = ints[0]['intent']
        for intent in self.user_data:
            if intent['tag'] == tag:
                result = intent['answer']
                break
        res = lambda y: y if y not in " " else "A"
        return res(result)

    def chatbot_response(self, msg):
        try:
            ints = self.predict_class(msg)
            res = self.get_response(ints)
        except:
            res = 'A'
        return res

class ChatbotAgent:
    def __init__(self,page,username):
        global user
        user = username
        self.page = page
        self.logger = logging.getLogger("jobautobot.naukri.chatbot")
        self.logger.info(f"Initializing ChatbotAgent for user: {username}")
        try:
            with open(f"./jab/data/{user}/training_data.json", 'r') as json_file:
                user_data = json.load(json_file)
                self.logger.debug(f"Loaded training data with {len(user_data)} intents")
            self.model = ChatbotModel(user_data)
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            self.logger.error(f"Error initializing ChatbotAgent: {e}")
            raise

    def sentiment_score(self,text):
        score = self.analyzer.polarity_scores(text)
        return score['compound'] 
    
    def match_by_sentiment(self,target, strings):
        target_sentiment = self.sentiment_score(target)
        sentiment_diffs = []
        for string in strings:
            string_sentiment = self.sentiment_score(string)
            diff = abs(target_sentiment - string_sentiment)
            sentiment_diffs.append((string, diff))
        
        best_match = min(sentiment_diffs, key=lambda x: x[1])
        return best_match[0], best_match[1]

    def classify_new_question(self):
        page=self.page
        self.logger.info("Waiting for chatbot message container")
        cbcn = page.wait_for_selector(".chatbot_MessageContainer", timeout=3000)
        time.sleep(2)
        try:
            while cbcn:
                question_element = page.locator(".botMsg").last
                self.page.wait_for_timeout(1000)
                
                question = question_element.inner_text()
                self.logger.info(f"New question appeared: {question}")
                answer = self.model.chatbot_response(question)
                self.logger.debug(f"Model generated answer: {answer}")

                # Log element detection
                checkboxes = cbcn.query_selector_all('input[type="checkbox"]')
                radio_buttons = cbcn.query_selector_all('input[type="radio"]')
                text_input = page.locator('.chatbot_MessageContainer .textArea')
                chip = page.query_selector('.chatbot_MessageContainer .chipsContainer .chatbot_Chip')
                suggs = cbcn.query_selector_all('.ssc__heading')
                dob = cbcn.query_selector(".dob__container")
                
                self.logger.debug(f"Form elements found - checkboxes: {len(checkboxes)}, radio_buttons: {len(radio_buttons)}, " +
                             f"text_input visible: {text_input.is_visible()}, chip: {bool(chip)}, " + 
                             f"suggestions: {len(suggs)}, dob: {bool(dob)}")
                
                if chip:
                    self.logger.info("Found chip option, clicking to skip")
                    chip.click()
                    continue
                elif radio_buttons or checkboxes:
                    _buttons = radio_buttons or checkboxes
                    self.logger.info(f"Question requires a radio/checkbox selection with {len(_buttons)} options")
                    options = [el.evaluate('el => el.id') for el in _buttons]
                    self.logger.debug(f"Options: {options}")
                    finnas = self.match_by_sentiment(answer, options)[0]
                    self.logger.info(f"Selected option: {finnas} based on sentiment matching")
                    label_ = page.locator(f'label[for="{finnas}"]')
                    label_.click(force=True)
                elif text_input.is_visible():
                    self.logger.info("Question requires text input")
                    text_input.type(answer, delay=100)
                elif suggs:
                    self.logger.info(f"Found {len(suggs)} suggestions")
                    options = [el.evaluate('el => el.innerText') for el in suggs]
                    self.logger.debug(f"Suggestion options: {options}")
                    finnas = self.match_by_sentiment(answer, options)[0]
                    self.logger.info(f"Selected suggestion: {finnas}")
                    page.click(f'text="{finnas}"')
                elif dob:
                    self.logger.info("Found date of birth input")
                    dob_parts = answer.strip().split("/")
                    self.logger.debug(f"Parsed DOB: {dob_parts}")
                    page.locator("input[name='day']").type(dob_parts[0], delay=100)
                    page.locator("input[name='month']").type(dob_parts[1], delay=100)
                    page.locator("input[name='year']").type(dob_parts[2], delay=100)
                else:
                    self.logger.warning("No recognized input elements found, returning")
                    return 
                
                send = page.locator('.sendMsg')
                try:
                    expect(send).to_be_enabled()
                    time.sleep(0.5)
                    self.logger.info("Clicking send button")
                    send.click(timeout=3000)
                except Exception as e:
                    self.logger.error(f"Error clicking send button: {e}")
                    return
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Error in classify_new_question: {e}")
            return {"response": 'error occured on classify_new_question', "error": str(e)}

class NaukriBot:
    def __init__(self, usreml, usrpas, username, number=10, debug=False):
        self.browser = None
        self.page = None
        self.usr = [usreml, usrpas]
        self.username = username
        self.applno = number
        self.applied_count = 0
        self.page_no = 1
        self.tabs = ["profile", "apply", "preference", "similar_jobs"]
        self.pattern = re.compile(r'https://.*/myapply/saveApply\?strJobsarr=')
        
        # Set up logger
        self.logger = logging.getLogger("jobautobot.naukri.bot")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Initializing NaukriBot for {username} to apply for {number} jobs")

    def close_popups(self):
        """Close any popup dialogs by clicking on elements with 'crossIcon' class"""
        try:
            print(f"[DEBUG] Checking for popup dialogs with 'crossIcon' class")
            self.logger.debug("Checking for popup dialogs with 'crossIcon' class")
            cross_icons = self.page.query_selector_all("div.crossIcon")
            print(f"[DEBUG] Found {len(cross_icons)} popup dialogs with 'crossIcon' class")
            if cross_icons:
                self.logger.info(f"Found {len(cross_icons)} popup dialogs to close")
                for i, icon in enumerate(cross_icons):
                    print(f"[DEBUG] Clicking popup close icon {i+1} of {len(cross_icons)}")
                    icon.click()
                    self.logger.debug(f"Clicked on popup close icon {i+1} of {len(cross_icons)}")
                    self.page.wait_for_timeout(500)  # Short pause between clicks
                print(f"[DEBUG] Successfully closed {len(cross_icons)} popups")
                return True
            print(f"[DEBUG] No popups with 'crossIcon' class found")
            return False
        except Exception as e:
            print(f"[ERROR] Error when trying to close popups: {e}")
            self.logger.warning(f"Error when trying to close popups: {e}")
            return False

    def init_browser(self):
        self.logger.info("Initializing browser")
        playwright = sync_playwright().start()
        args = ["--disable-blink-features=AutomationControlled"]
        
        # Add more detailed debugging for browser launch
        try:
            self.logger.debug("Launching Chromium browser")
            self.browser = playwright.chromium.launch(headless=False, args=args)
            self.logger.debug("Creating new page")
            self.page = self.browser.new_page()
            self.logger.debug("Initializing ChatbotAgent")
            self.cba = ChatbotAgent(self.page, self.username)
            self.logger.info("Browser initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            raise

    def login(self):
        print(f"[DEBUG] Starting login process")
        self.logger.info("Starting login process")
        try:
            print(f"[DEBUG] Navigating to Naukri.com")
            self.logger.debug("Navigating to Naukri.com")
            self.page.goto("http://www.naukri.com", timeout=40000)            
            
            print(f"[DEBUG] Checking for popups after navigation")
            # Close any popups that might appear on the login page
            self.close_popups()
            
            print(f"[DEBUG] Clicking login button")
            self.logger.debug("Clicking login button")
            self.page.click('//*[@id="login_Layer"]')
            print(f"[DEBUG] Entering email: {self.usr[0]}")
            self.logger.debug(f"Entering email: {self.usr[0]}")
            self.page.type('input[type="text"]', self.usr[0], delay=100)
            print(f"[DEBUG] Entering password")
            self.logger.debug("Entering password")
            self.page.type('input[type="password"]', self.usr[1], delay=100)
            print(f"[DEBUG] Clicking submit button")
            self.logger.debug("Clicking submit button")
            self.page.click('button[type="submit"]')
            
            try:
                print(f"[DEBUG] Waiting for homepage to load...")
                self.logger.info("Waiting for homepage to load...")
                self.page.wait_for_url("https://www.naukri.com/mnjuser/homepage", timeout=30000)
                print(f"[DEBUG] URL matched, waiting for page to be stable...")
                self.logger.info("URL matched, now waiting for page to be stable...")
                self.page.wait_for_load_state("networkidle", timeout=30000)
                print(f"[DEBUG] Login successful. Current URL: {self.page.url}")
                self.logger.info(f"Login successful. Current URL: {self.page.url}")
                
                # Take a screenshot after successful login
                self.logger.debug("Taking screenshot after login")
                screenshot_path = f"./naukri_login_success_{int(time.time())}.png"
                self.page.screenshot(path=screenshot_path)
                print(f"[DEBUG] Screenshot saved to {screenshot_path}")
                self.logger.debug(f"Screenshot saved to {screenshot_path}")
                
                # Wait for an additional 5 seconds to ensure page is fully loaded
                self.page.wait_for_timeout(5000)
                return True
            except Exception as e:
                print(f"[ERROR] Login navigation failed: {e}")
                self.logger.warning(f"Login navigation failed: {e}")
                print(f"[DEBUG] Current URL: {self.page.url}")
                self.logger.info(f"Current URL: {self.page.url}")
                
                # Take screenshot of whatever page we landed on
                screenshot_path = f"./naukri_login_failed_{int(time.time())}.png"
                self.page.screenshot(path=screenshot_path)
                print(f"[DEBUG] Failure screenshot saved to {screenshot_path}")
                self.logger.debug(f"Failure screenshot saved to {screenshot_path}")
                
                # Try to collect and log any error messages on the page
                try:
                    error_elements = self.page.query_selector_all(".error-message, .alert, .notification")
                    for i, el in enumerate(error_elements):
                        print(f"[ERROR] Error message {i + 1}: {el.inner_text()}")
                        self.logger.warning(f"Error message {i + 1}: {el.inner_text()}")
                except Exception:
                    pass
                
                # Still return True if we're on any Naukri page after login attempt
                if "naukri.com" in self.page.url:
                    print(f"[DEBUG] Still on Naukri site, continuing...")
                    self.logger.info("Still on Naukri site, continuing...")
                    return True
                print(f"[ERROR] Login failed.")
                self.logger.error("Login failed.")
                return False
        except Exception as e:
            print(f"[ERROR] Error during login: {e}")
            self.logger.error(f"Error during login: {e}")
            return False
        
    def checkbox_apply(self):
        try:
            checkboxes = self.page.locator('.naukicon-ot-checkbox').element_handles()
            self.logger.info(f"Found {len(checkboxes)} checkboxes.")
            if not len(checkboxes)==0:          
                lcbxs = 0
                for checkbox in checkboxes[:5]:
                    checkbox.click()
                    lcbxs += 1                
                apply_button = self.page.locator('.multi-apply-button')
                apply_button.click()
                try:
                    expect(self.page.locator(".chatbot_MessageContainer")).to_be_visible(timeout=3000)
                except:
                    try:
                        expect(self.page).to_have_url(self.pattern)
                        return {"status":"done","clicked":lcbxs}
                    except:
                        raise
                return {"status":"underway","found":len(checkboxes),"clicked":lcbxs}
            else:
                return {"status":"finished","found":len(checkboxes),"clicked":0}
        except Exception:
            return {"status":"failed"}
        
    def apply_(self):
        print(f"[DEBUG] Starting apply_ method")
        self.logger.debug("Starting apply_ method")
        self.page.wait_for_load_state('networkidle')
        
        # Close any popups before getting job links
        print(f"[DEBUG] Checking for popups before getting job links")
        self.close_popups()
        
        print(f"[DEBUG] Getting job links from page")
        job_links = self.page.eval_on_selector_all(
            '.title',
            'elements => elements.map(element => element.getAttribute("href")) .filter(href => href !==null)'
        )
        print(f"[DEBUG] Found {len(job_links)} job links on current page")
        self.logger.info(f"Found {len(job_links)} job links on current page")
        
        for i, jl in enumerate(job_links):
            if self.applied_count >= self.applno:
                print(f"[DEBUG] Reached target number of applications ({self.applno})")
                self.logger.info(f"✅ Applied to {self.applied_count} jobs.")
                break
                
            print(f"[DEBUG] Processing job link {i+1}/{len(job_links)}: {jl}")
            self.logger.debug(f"Processing job link {i+1}/{len(job_links)}")
            try:
                print(f"[DEBUG] Waiting before navigating to job page")
                self.page.wait_for_timeout(2000)
                print(f"[DEBUG] Navigating to job page: {jl}")
                self.page.goto(jl)
                print(f"[DEBUG] Waiting for page to load")
                # self.page.wait_for_load_state('networkidle')
                
                # Close any popups before clicking apply button
                print(f"[DEBUG] Checking for popups before clicking apply button")
                self.close_popups()
                
                print(f"[DEBUG] Looking for apply button")
                apply = self.page.query_selector('#apply-button')
                if apply:
                    print(f"[DEBUG] Found apply button, clicking it")
                    apply.click()
                    print(f"[DEBUG] Checking for chatbot message container")
                    try:
                        expect(self.page.locator(".chatbot_MessageContainer")).to_be_visible(timeout=3000)
                        print(f"[DEBUG] Chatbot detected, starting conversation")
                        self.logger.info("Chatbot detected, starting conversation")
                        self.cba.classify_new_question()
                        self.applied_count+=1
                        print(f"[DEBUG] Application submitted via chatbot. Total applications: {self.applied_count}")
                        self.logger.info(f"Application submitted via chatbot. Total: {self.applied_count}/{self.applno}")
                    except Exception as e:
                        print(f"[DEBUG] Chatbot not detected, checking for pattern URL: {e}")
                        try:
                            expect(self.page).to_have_url(self.pattern)
                            self.applied_count+=1
                            print(f"[DEBUG] Application submitted via direct apply. Total applications: {self.applied_count}")
                            self.logger.info(f"Application submitted via direct apply. Total: {self.applied_count}/{self.applno}")
                        except Exception as e2:
                            print(f"[WARNING] Could not detect successful application: {e2}")
                            self.logger.warning(f"Could not detect successful application: {e2}")
                else:
                    print(f"[WARNING] Apply button not found on job page")
                    self.logger.warning("Apply button not found on job page")
            except Exception as e:
                print(f"[ERROR] Error applying to job {jl}: {e}")
                self.logger.warning(f"Error applying to job {jl}: {e}")
                continue 
                
        if self.applied_count < self.applno:
            self.page_no += 1
            print(f"[DEBUG] Need more applications, going to page {self.page_no}")
            self.logger.info(f"Going to page {self.page_no} for more jobs")
            
            parsed = urlparse(self.base_page_url)
            new_path = parsed.path + f"-{self.page_no}"
            modified_url = urlunparse(parsed._replace(path=new_path))
            
            try:
                print(f"[DEBUG] Navigating to next page: {modified_url}")
                self.page.goto(modified_url)
            except Exception as e:
                print(f"[DEBUG] No more pages available: {e}")
                self.logger.info(f"✅ Applied to {self.applied_count} jobs. No more pages available.")
                return
                
            self.logger.info(f'Going to page {self.page_no}')
            self.apply_()

    def filter_apply(self,s,e='',l='',ja='3'):
        self.search = s
        if not self.search:
            self.logger.warning("Search keyword required")
            return
        self.experience = e
        self.location = l
        self.jobage = ja
        self.init_browser()
        self.login()
        time.sleep(1)
        self.filter_()
        self.base_page_url = self.page.url
        self.apply_()
        self.page.close()
        return {"response":"applied successfully","applied":self.applied_count}

    def filter_(self):
        serch = self.page.locator(".nI-gNb-sb__icon-wrapper")
        serch.click() 
        self.page.locator('input[placeholder="Enter keyword / designation / companies"]').type(self.search,delay=100)
        if self.location:
            self.page.locator('input[placeholder="Enter location"]').type(self.location,delay=100)
        if self.experience:
            self.page.locator('#experienceDD').click()
            self.page.locator(f'li[index="{self.experience}"]').click()
        serch.click()
        self.page.wait_for_load_state('load')
        # Close any popups that might appear after filtering
        self.close_popups()
        
        curl = self.page.url 
        if self.jobage:
            nurl = curl+f"&jobAge={self.jobage}"
            self.page.goto(nurl)

    def start_apply(self,tab):
        self.tabIndex = 0
        self.tab = tab
        self.init_browser()
        if self.login():
            print(f"[DEBUG] Bot action started for tab: {self.tab}")
            botactions = self.bot_actions()
            return botactions
        
    def bot_actions(self):
        try:
            print(f"[DEBUG] Starting bot_actions method with tab: {self.tab}")
            self.logger.info(f"Starting bot_actions with tab: {self.tab}")
            time.sleep(2)
            
            # Close any popups before clicking menu items
            print(f"[DEBUG] Checking for popups before clicking menu items")
            self.close_popups()
            
            print(f"[DEBUG] Clicking on menu dropdown")
            self.logger.debug("Clicking on menu dropdown")
            self.page.click('.nI-gNb-menuItems__anchorDropdown')
            
            if not self.tab=="profile":
                print(f"[DEBUG] Clicking on tab: {self.tab}")
                self.logger.debug(f"Clicking on tab: {self.tab}")
                self.page.click(f"#{self.tab}")
            
            print(f"[DEBUG] Waiting for page to load")
            self.logger.debug("Waiting for page to load")
            # self.page.wait_for_load_state("networkidle")
            print(f"[DEBUG] Current URL: {self.page.url}")
            
            if self.applied_count >= self.applno:
                print(f"[DEBUG] Reached application target: {self.applied_count}/{self.applno}")
                self.logger.info(f"applied {self.applied_count} jobs")
                return {"response":"applied successfully","applied":self.applied_count}
            else:
                print(f"[DEBUG] Current application count: {self.applied_count}/{self.applno}")
                print(f"[DEBUG] Attempting checkbox_apply")
                self.logger.debug("Attempting checkbox_apply")
                cbapl = self.checkbox_apply()
                print(f"[DEBUG] checkbox_apply result: {cbapl}")
                self.logger.debug(f"checkbox_apply result: {cbapl}")
            
            if cbapl["status"] == 'failed':
                print(f"[DEBUG] Daily quota finished with {self.applied_count} applications")
                self.logger.info(f"finished daily quota with {self.applied_count} jobs")
                self.close()
                return {"response":"quota finished","applied":self.applied_count}
            elif cbapl["status"] == 'done':
                print(f"[DEBUG] Applications completed, clicked: {cbapl['clicked']}")
                self.logger.debug(f"Applications completed, clicked: {cbapl['clicked']}")
                self.applied_count += cbapl["clicked"]
                print(f"[DEBUG] Updated application count: {self.applied_count}/{self.applno}")
                self.logger.info(f"Updated application count: {self.applied_count}/{self.applno}")
                self.bot_actions()
            elif cbapl["status"] == 'underway':
                print(f"[DEBUG] Application in progress, starting chatbot interaction")
                self.logger.debug("Application in progress, starting chatbot interaction")
                self.cba.classify_new_question()
                try:
                    print(f"[DEBUG] Checking for completion pattern in URL")
                    self.logger.debug("Checking for completion pattern in URL")
                    expect(self.page).to_have_url(self.pattern)
                    print(f"[DEBUG] URL pattern matched, applications successful: {cbapl['clicked']}")
                    self.logger.info(f"URL pattern matched, applications successful: {cbapl['clicked']}")
                    self.applied_count += cbapl["clicked"]
                    print(f"[DEBUG] Updated application count: {self.applied_count}/{self.applno}")
                    self.logger.info(f"Updated application count: {self.applied_count}/{self.applno}")
                    self.bot_actions()
                except Exception as e:
                    print(f"[ERROR] Error answering Naukri questions: {e}")
                    self.logger.error(f"An error occured answering naukri questions :===>{e}")
                    print(f"[DEBUG] Taking screenshot of error state")
                    screenshot_path = f"./naukri_error_{int(time.time())}.png"
                    self.page.screenshot(path=screenshot_path)
                    print(f"[DEBUG] Error screenshot saved to {screenshot_path}")
                    self.close()
                    return {"response":"error on botactions","error":str(e)}
            elif cbapl['status']=="finished":
                print(f"[DEBUG] Current tab finished, switching to next tab")
                self.logger.debug("Current tab finished, switching to next tab")
                self.tabIndex += 1
                if self.tabIndex < len(self.tabs):
                    self.tab = self.tabs[self.tabIndex]
                    print(f"[DEBUG] Moving to next tab: {self.tab}")
                    self.logger.info(f"Moving to next tab: {self.tab}")
                    self.bot_actions()
                else:
                    print(f"[DEBUG] All tabs processed, completing with {self.applied_count} applications")
                    self.logger.info(f"All tabs processed, completing with {self.applied_count} applications")
                    self.close()
                    return {"response":"all tabs completed","applied":self.applied_count}
        except Exception as e:
            print(f"[ERROR] Exception in bot_actions: {e}")
            self.logger.error(f"applied {self.applied_count} jobs but an error occured :===>{str(e)}")
            print(f"[DEBUG] Taking screenshot of error state")
            try:
                screenshot_path = f"./naukri_error_{int(time.time())}.png"
                self.page.screenshot(path=screenshot_path)
                print(f"[DEBUG] Error screenshot saved to {screenshot_path}")
            except Exception as screenshot_error:
                print(f"[ERROR] Failed to take error screenshot: {screenshot_error}")
            self.close()

    def close(self):
        """Close the browser properly and clean up resources"""
        try:
            print(f"[DEBUG] Closing browser")
            self.logger.info("Closing browser")
            if self.page:
                self.page.close()
                print(f"[DEBUG] Page closed")
            if self.browser:
                self.browser.close()
                print(f"[DEBUG] Browser closed")
            print(f"[DEBUG] Cleanup complete")
            self.logger.info("Browser closed successfully")
        except Exception as e:
            print(f"[ERROR] Error closing browser: {e}")
            self.logger.error(f"Error during browser closure: {e}")
