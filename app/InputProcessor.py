import TopicModellingService as topicModellingService
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
import dill as pickle
import re
import string
from datetime import datetime
import calendar
import numpy as np


def process_and_validate_input(args, result):
    requiredFields = ["group", "name", "description", "time"]
    optionalFields = ["venue", "duration", "rsvp_limit"]

    for field in requiredFields:
        if args.get(field) is None:
            return False
        if field == "time" and args.get(field, type=int) > 0:
            result[field] = args.get(field, type=int)
        else:
            result[field] = args.get(field)
            if len(result[field]) <= 0:
                return False

    for field in optionalFields:
        data = args.get(field, type=int)
        if data is None:
            continue
        result[field] = data

    return True


def transform_input(params, modelAccessor):
    # Get the model parameter list
    with open('../dicts/Column_List.pkl', 'rb') as handle:
        columns_list = pickle.load(handle)
    handle.close()

    # Create an empty dictionary from the model parameter list
    d = dict.fromkeys(columns_list)

    transform_event_day(columns_list, params, d)
    transform_event_hour_and_notice(columns_list, params, d)
    transform_venue_and_group(columns_list, params, d)
    transform_event_desc_and_name(columns_list, params, d, modelAccessor)
    transform_duration(params, d)
    transform_rsvp_limit(params, d)

    # Convert filled dictionary to dataframe since model is trained on pandas dataframe.
    result = pd.DataFrame(d, index=['i']).reset_index().drop('index', 1)

    return result

# Function to obtain and transform event_day
def transform_event_day(columns_list, params, d):
    filter_col_eventDay = [col for col in columns_list if col.startswith('Event_Day_')]
    # Get the day of the event from the epoch timestamp
    Event_Day = datetime.fromtimestamp(params['time'] / 1000).strftime("%A")
    # Change the name of the days to number of the days. (Monday -->0,...)
    days = dict(zip(calendar.day_name, range(7)));
    Event_Day = int(days[Event_Day])

    # One-Hot Encoding of Event_Day
    categorical_fix("Event_Day", Event_Day, d, filter_col_eventDay)

# Function to obtain and transform event_hour and event_notice
def transform_event_hour_and_notice(columns_list,params, d):
    filter_col_eventHour = [col for col in columns_list if col.startswith('Event_Hour_')]
    # Get the hour of the event from the epoch timestamp
    Event_Hour = int(datetime.fromtimestamp(params['time'] / 1000).strftime("%H"))

    # One-Hot Encoding of Event_Hour
    categorical_fix("Event_Hour", Event_Hour, d, filter_col_eventHour)

    params['time'] = datetime.strptime(
        datetime.fromtimestamp(params['time'] / 1000).strftime('%m/%d/%y %H:%M:%S'),
        '%m/%d/%y %H:%M:%S')
    now = datetime.strptime(datetime.now().strftime('%m/%d/%y %H:%M:%S'),
                            '%m/%d/%y %H:%M:%S')
    # Calculate Event Notice by calculating the difference between event time and event creation time
    if params['time'] > now:
        d['Event_Notice'] = np.log1p(int((params['time'] - now).total_seconds() / 3600))
    else:
        d['Event_Notice'] = np.log1p(24)

# Function to transform venue and group id
def transform_venue_and_group(columns_list, params, d):
    filter_col_group = [col for col in columns_list if col.startswith('group_id_')]
    filter_col_venue = [col for col in columns_list if col.startswith('venue_id_')]

    group_id = params['group']
    # Get total user count of the given group from pre-created dictionary
    with open('../dicts/Group_UserCount.pkl', 'rb') as group_user_count:
        group_userCount = pickle.load(group_user_count)
        d['user_count'] = np.log1p(group_userCount.get('user_count').get(group_id))
    group_user_count.close()

    # venue_id is optional so if its not given, we use the most frequent venue for the given group.
    if 'venue_id' in params and params['venue'] > 0:
        venue_id = int(params['venue'])
    else:
        with open('../dicts/Group_Value.pkl', 'rb') as handle:
            group_venue_list = pickle.load(handle)
        venue_id = group_venue_list.get('venue_id').get(group_id)
        handle.close()

    # One-Hot Encoding of venue_id
    categorical_fix("venue_id", venue_id, d, filter_col_venue)
    # One-Hot Encoding of group_id
    categorical_fix("group_id", group_id, d, filter_col_group)

# Function to transform event description and event name
def transform_event_desc_and_name(columns_list, params, d, modelAccessor):
    filter_col_dom_topic = [col for col in columns_list if col.startswith('Dominant_Topic_')]
    filter_col_dom_topic_eventName = [col for col in columns_list if col.startswith('Dominant_Topic_Event_Name_')]

    # Clean event description and event name from all the html, javascript tags.
    Event_Description = clean_description(params['description'])
    Event_Name = clean_description(params['name'])

    # With the topic modelling, obtain the dominant topic number for the given event description and event name
    Dominant_Topic = float(topicModellingService.topic_modelling(Event_Description, modelAccessor.lda_model))
    Dominant_Topic_Event_Name = float(topicModellingService.topic_modelling(Event_Name, modelAccessor.lda_model_eventName))

    # One-Hot Encoding of Dominant_Topic and Dominant_Topic_Event_Name
    categorical_fix("Dominant_Topic", Dominant_Topic, d, filter_col_dom_topic)
    categorical_fix("Dominant_Topic_Event_Name", Dominant_Topic_Event_Name, d, filter_col_dom_topic_eventName)

    # Obtain event description length
    d['Event_Description_Length'] = np.log1p(len(Event_Description.split()))


def transform_duration(params, d):
    # Duration is optional parameter if its not given then assign the mean duration of events from training data
    if 'duration' in params and params['duration'] > 0:
        d['duration'] = np.log1p((int(params['duration'] / (1000 * 60 * 60)) % 24))
    else:
        d['duration'] = np.log1p(6)  # mean duration of events (h)


def transform_rsvp_limit(params, d):
    # rsvp_limit is optional parameter if its not given then assign 999 meaning no limit.
    if 'rsvp_limit' in params and params['rsvp_limit'] > 0:
        d['rsvp_limit'] = int(params['rsvp_limit'])
    else:
        d['rsvp_limit'] = 999


def categorical_fix(field_name, field_value, d, filtered_list):
    if (field_name + "_" + str(field_value)) in filtered_list:
        d[(field_name + "_" + str(field_value))] = 1
        for i in [n for n in filtered_list if n != (field_name + "_" + str(field_value))]:
            d[str(i)] = 0
    return d


# Function to clean the description field. Clean description will be used as an input for the topic detection
def clean_description(decription_text):
    text_string = BeautifulSoup(decription_text, "lxml").text
    # remove numbers
    text_nonum = re.sub(r'\d+', '', unicodedata.normalize("NFKD", text_string))
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
    # remove url
    text_no_url = re.sub('(http|https)\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?', '', text_nopunct,
                         flags=re.MULTILINE)
    # REMOVE BULLET POINTS
    text_no_bullet = re.sub('\d\.\s+|[a-z]\)\s+|•\s+|[A-Z]\.\s+|[IVX]+\.\s+', ' ', text_no_url)
    # remove emails
    text_no_email = re.sub('\S*@\S*\s?', '', text_no_bullet)
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub('\s+', ' ', text_no_email).strip()
    return text_no_doublespace
