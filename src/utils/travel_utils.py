from openai import OpenAI
import json
import os
import logging
from utils.flow_utils import get_response_from_client

def convert_to_json_with_gpt(text, openai_key, max_fail=3, model_name='gpt-4-1106-preview'):
    todo_prompt = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
            -----EXAMPLE-----
            [{
                    "days": 1,
                    "current_city": "from Dallas to Peoria",
                    "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
                    "breakfast": "-",
                    "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
                    "lunch": "-",
                    "dinner": "Tandoor Ka Zaika, Peoria",
                    "accommodation": "Bushwick Music Mansion, Peoria"
                },
                {
                    "days": 2,
                    "current_city": "Peoria",
                    "transportation": "-",
                    "breakfast": "Tandoor Ka Zaika, Peoria",
                    "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
                    "lunch": "Cafe Hashtag LoL, Peoria",
                    "dinner": "The Curzon Room - Maidens Hotel, Peoria",
                    "accommodation": "Bushwick Music Mansion, Peoria"
                },
                {
                    "days": 3,
                    "current_city": "from Peoria to Dallas",
                    "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "-",
                    "accommodation": "-"
                }]
            -----EXAMPLE END-----
            """

    client = OpenAI(api_key=openai_key)

    total_price = 0.0
    attempt = 1
    while attempt <= max_fail:
        prompt = todo_prompt + f"text: {text}\njson:"
        response, price = get_response_from_client(client, [{'role': 'user', 'content': prompt}], model_name, 1.)
        total_price += price

        logging.info((f'Generated JSON: \n```\n{response}\n```'))

        try:
            result = response.split('```json')[1].split('```')[0]
        except:
            attempt += 1
            todo_prompt += f"Previous generated plan: {response}\nThis plan cannot be parsed. The plan has to follow the format ```json [The generated json format plan]```\n"
            continue

        try:
            result = eval(result)
        except:
            attempt += 1
            todo_prompt += f"Previous generated plan: {response}\nThis is an illegal json format.\n"
        
        break
    
    if attempt > max_fail:
        result = None

    return result, total_price


def get_result_file(args):
    result_file = os.path.join(args.results_dir, args.task, f"{args.set_type}_{args.model_name.replace('/','_')}_{args.get_observation}_{args.results_name}.jsonl")
    if not os.path.exists(os.path.join(args.results_dir, args.task)):
        os.makedirs(os.path.join(args.results_dir, args.task))
    return result_file

def write_result_into_file(result, result_file):
    with open(result_file, 'a') as w:
        output = json.dumps(result)
        w.write(output + '\n')
        w.close()
    return