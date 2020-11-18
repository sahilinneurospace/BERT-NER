import json
from utils import DateTimeParser

## Entity extraction function to extract all entities (compatible with B/I labelling scheme), along with entity order and line-wise placement information required for task generation
def extract_entities(seq_in, ner_output, coordinates):
	seq_in = seq_in.split()
	ner_output = ner_output.split()
	curr_entity = ner_output[0]
	
	if curr_entity[:2] == "B_":
		curr_entity = curr_entity[2:]

	# exhaustive list of entities currently in use
	entity_list = ['Movie_name', 'Book_name', 'Genre', 'Person_details', 'Product_details', 'ToDo', 'O', 'Organization_name', 'Role_in_organization', 'Quantity', 'Email', 'Phone', 'Product_name', 'Time', 'Event', 'Date', 'App_name', 'Address', 'Fax_number', 'Person_name', 'Details',  'Url', 'Item', 'Subevent']
	
	# dictionary of entities containing entity name, its order in the sequence of text, the lines it is spread across, info used for task generation
	extracted_entities = {}
	for entity in entity_list:
		extracted_entities[entity] = []
	entity = []
	lines = []
	line = 1
	order = 1

	for i in range(len(ner_output)):
		if i > 0 and coordinates[i][1] > coordinates[i-1][3]:
			line += 1
		if ner_output[i] == curr_entity :
			if curr_entity != 'O':
				entity.append(seq_in[i])
				if line not in lines:
					lines.append(line)
		else:
			if ner_output[i][:2] == "B_" or (ner_output[i][:2] != "B_" and not extracted_entities[ner_output[i]]):
				if curr_entity != 'O':
					extracted_entities[curr_entity].append([entity, lines, order])
				entity = [seq_in[i]]
				lines = [line]
				curr_entity = ner_output[i]
				order += 1
				if ner_output[i][:2] == "B_":
					curr_entity = ner_output[i][2:]
			else:
				extracted_entities[ner_output[i]][-1][0].append(seq_in[i])
				if line not in extracted_entities[ner_output[i]][-1][1]:
					extracted_entities[ner_output[i]][-1][1].append(line)
	if curr_entity != 'O':
		extracted_entities[curr_entity].append([entity, lines, order])
	for entity in extracted_entities.keys():
		extracted_entities[entity] = [[" ".join(x[0]), x[1], x[2]] for x in extracted_entities[entity]]
	return extracted_entities

# Task generation method using entity information extracted
def gen_task_card(entities, intent):
	cards = []
	if intent == 'Contacts':
		taskdict = {}
		taskdict["Title"] = "Add to contacts"
		taskdict["Due Date"] = "NA"
		taskdict["Reminder Date"] = "NA"
		taskdict["Details"] = {}
		if entities['Person_name']:
			taskdict["Details"]["Name"] = entities['Person_name'][0][0]
		if entities['Role_in_organization']:
			taskdict["Details"]["Organization"] = entities['Role_in_organization'][0][0]
		if entities['Organization_name']:
			taskdict["Details"]["Organization"] = [entity[0] for entity in entities['Organization_name']]
		if entities['Phone_number']:
			taskdict["Details"]["Phone_number"] = [entity[0] for entity in entities['Phone_number']]
		if entities['Fax_number']:
			taskdict["Details"]['Fax_number'] = [entity[0] for entity in entities['Fax_number']]
		if entities['Email_id']:
			taskdict["Details"]['Email_id'] = [entity[0] for entity in entities['Email_id']]
		if entities['Url']:
			taskdict["Details"]['Url'] = [entity[0] for entity in entities['Url']]
		cards.append(taskdict)
	if intent == 'Event':
		events = []
		for i in range(len(entities['Event_name'])):
			taskdict = {}
			if i < len(entities['Date']):
				taskdict["DueDateTime"] = {"DateTime": DateTimeParser(entities["Date"][i][0], "")[0], "TimeZone": ""}
			else:
				taskdict["DueDateTime"] = {"DateTime": "", "TimeZone": ""}
			taskdict["Body"] = {"Content": "Add reminder for " + entities['Event_name'][i][0] + "\n"}
			taskdict["Body"]["ContentType"] = "Text"
			taskdict["ReminderDateTime"] = ""
			taskdict["Subject"] = entities['Event_name'][i][0]
			if i < len(entities['Time']):
				taskdict["DueDateTime"]["DateTime"] += " " + DateTimeParser("" ,entities["Time"][i][0])[1]
			if i < len(entities['Address']):
				taskdict["Body"]["Content"] += "Venue: " + entities["Address"][i][0] + "\n"
			if i < len(entities['Event_details']):
				taskdict["Body"]["Content"] += "Details: " + entities["Event_details"][i][0] + "\n"
			events.append(taskdict)
		cards.append({"Title": "Create reminder for " + entities['Event_name'][i][0], 
			"Actions": [{"Action": "Create reminder", "TaskFolderId": "Events", "Tasks": events}]})
	if intent == 'Shopping(NonGrocery)':
		names = entities['Product_name']
		details = entities['Product_details']
		products = {}
		for name in names:
			products[name[0]] = []
		if details and details[0][2] < names[0][2]:
			products[names[0][0]].append(details[0][0])
			details = details[1:]
		j = 0
		for i in range(len(details)):
			while j < len(names) - 1 and details[i][2] > names[j+1][1]:
				j += 1
			products[names[j][0]].append(details[i][0])
		shopping_bag = []
		for name in products.keys():
			taskdict = {}
			taskdict["Body"] = {"Content": "Buy " + name + "\nDetails:\n " + "\n".join(products[name]), "ContentType": "Text"}
			taskdict["Subject"] = name
			taskdict["DueDateTime"] = {"DateTime": "", "TimeZone": ""}
			taskdict["ReminderDateTime"] = ""
			shopping_bag.append(taskdict)
		taskdict = {"Title": "Add " + str(len(shopping_bag)) + " items to Shopping list", 
			"Actions": [{"Action": "Add to Shopping list", "TaskFolderId": "Shopping(NonGrocery)", "Tasks": shopping_bag}]}
		cards.append(taskdict)
	if intent == 'Shopping(Grocery)':
		items = entities['Item']
		shopping_list = {item[0]: [] for item in items}
		qties = entities['Quantity']
		# qty_assignment contains the mapping between each quantity and
		# the item it is associated with. ith element in this list
		# would be the index of the item associated with the ith quantity
		qty_assignments = []
		for qty in qties:
			items_ = [[i]+items[i] for i in range(len(items))]
			# quantity entity will be matched to the item entity occuring closest
			# to it in the order of entities. If there are multiple such items, one 
			# sharing the same line 
			items_ = [item for item in items_ if abs(item[3] - qty[2]) == min([abs(x[3] - qty[2]) for x in items_])]
			if len(items_) == 1:
				qty_assignments.append(items_[0][0])
			else:
				# intersection of lines the quantity is spread across and the item is spread across used to filter items
				if [item for item in items_ if list(set(item[2]) & set(qty[1]))]:
					items_ = [item for item in items_ if list(set(item[2]) & set(qty[1]))]
				qty_assignments.append(items_[0][0])
		for i, j in enumerate(qty_assignments):
			shopping_list[items[j][0]].append(qties[i][0])
		shopping_bag = []
		for item in shopping_list.keys():
			taskdict = {}
			taskdict["Body"] = {}
			taskdict["Subject"] = item
			taskdict["DueDateTime"] = {"DateTime": "", "TimeZone": ""}
			taskdict["Body"]["Content"] = "Buy " + " ".join(shopping_list[item]) + " " + item
			taskdict["Body"]["ContentType"] = "Text"
			shopping_bag.append(taskdict)
		taskdict = {"Title": "Add " + str(len(items)) + " items to Grocery list", 
			"Actions": [{"Action": "Add to Grocery list", "TaskFolderId": "Grocery", "Tasks": shopping_bag}]}
		cards.append(taskdict)
	if intent == 'Collections(Books)':
		books = [x[0] for x in entities['Book_name']]
		books_ = books[:]
		auths = [x[0] for x in entities['Person_name']]
		genrs = [x[0] for x in entities['Genre']]
		book_list = []
		while books:
			taskdict = {}
			taskdict["Body"] = {"Content": books[0] + " notes", "ContentType": "Text"}
			taskdict["Subject"] = books[0]
			taskdict["ReminderDateTime"] = ""
			taskdict["DueDateTime"] = {"DateTime": "", "TimeZone": ""}
			if genrs or auths:
				taskdict["Body"]["Content"] += "\n"
			if genrs:
				taskdict["Body"]["Content"] += "Genre: " + genrs[0] + "\n"
				genrs = genrs[1:]
			if auths:
				taskdict["Body"]["Content"] += "Author: " + auths[0]
				auths = auths[1:]
			books = books[1:]
			book_list.append(taskdict)
		taskdict = {"Title": "Add " + ", ".join(books_) + " to your ToDo list", 
			"Actions": [{"Action": "Add to Reading list", "TaskFolderId": "Reading", "Tasks": book_list}, 
					{"Action": "Add to Shopping list", "TaskFolderId": "Shopping", "Tasks": book_list}]}
		cards.append(taskdict)
	if intent == 'Collections(Movies)':
		movies = [x[0] for x in entities['Movie_name']]
		genrs = [x[0] for x in entities['Genre']]
		movie_list = []
		while movies:
			taskdict = {}
			taskdict["Body"] = {"Content": movies[0] + " notes", "ContentType": "Text"}
			taskdict["Subject"] = movies[0]
			taskdict["ReminderDateTime"] = ""
			taskdict["DueDateTime"] = {"DateTime": "", "TimeZone": ""}
			if genrs:
				taskdict["Body"]["Content"] += "\nGenre: " + genrs[0] + "\n"
				genrs = genrs[1:]
			movies = movies[1:]
			movie_list.append(taskdict)
		taskdict = {"Title": "Add to your Movie list", 
			"Actions": [{"Action": "Add to Movie list", "TaskFolderId": "Movies", "Tasks": movie_list}]}
		cards.append(taskdict)
	return cards


def gen_titles(entities):

	cards = []
	
	if entities['Person_name'] and (entities["Phone_number"] or entities["Fax_number"] or entities["Email_id"]):
		cards.extend(gen_task_card(entities, 'Contacts'))
	if entities['Event_name'] or entities['Event_details']:
		cards.extend(gen_task_card(entities, 'Event'))
	if entities['App_name']:
		cards.extend(gen_task_card(entities, 'Apps'))
	if entities['ToDo']:
		cards.extend(gen_task_card(entities, 'ToDo'))
	if entities['Product_name']:
		cards.extend(gen_task_card(entities, 'Shopping(NonGrocery)'))
	if entities['Item']:
		cards.extend(gen_task_card(entities, 'Shopping(Grocery)'))
	if entities['Book_name']:
		cards.extend(gen_task_card(entities, 'Collections(Books)'))
	if entities['Movie_name']:
		cards.extend(gen_task_card(entities, 'Collections(Movies)'))
	string = json.dumps({"Cards": cards}, indent=4) + "\n"
	return string


if __name__ == "__main__":

	file = open("data/train/seq_in.txt", 'r', encoding="utf-8")
	seq_in = file.read().split('\n')
	file.close()

	file = open("data/train/seq_out.txt", 'r')
	seq_out = file.read().split('\n')
	file.close()

	file = open("data/train/labels.txt", 'r')
	intents = file.read().split('\n')
	file.close()

	file = open("data/train/coordinates.txt", 'r')
	coordinates = file.read()
	coordinates = [[[float(x) for x in coord.split()] for coord in coords.split('\n')] for coords in coordinates.split('\n---------\n')]
	file.close()

	for i in range(len(seq_in)):
		seq, nero, intent, coords = seq_in[i], seq_out[i], intents[i], coordinates[i]
		print("\n\n", seq)
		entities = extract_entities(seq, nero, coords)
		#print(entities)
		string = gen_task_card(entities)
		print(string)
