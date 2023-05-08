from numpy import extract


all_domains = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police"]
db_domains = ['restaurant', 'hotel', 'attraction', 'train']
# normalize slot names


informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
all_infslot = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
                     "leave", "destination", "departure", "arrive", "department", "food", "time"]


normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "leaveat": "leave",
    "arriveby": "arrive",
    "price range": "pricerange",
    "trainid": "id",
    "addr": "address",
    "fee": "price",
    "post": "postcode",
    "ref": "reference",
    "ticket": "price",
    "depart": "departure",
    "dest": "destination"
}

req_slots = {
    "taxi": ["car", "phone"],
    "train": ["duration", "leave", "price", "arrive", "id"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area"],
    "hospital": ["address", "postcode", "phone"],
    "police": ["address", "postcode", "phone"]
}

all_req_slots = [
    "taxi-car",
    "taxi-phone",
    "train-duration",
    "train-leave",
    "train-price",
    "train-arrive",
    "train-id",
    "restaurant-phone",
    "restaurant-postcode",
    "restaurant-address",
    "restaurant-pricerange",
    "restaurant-food",
    "restaurant-area",
    "hotel-address",
    "hotel-postcode",
    "hotel-internet",
    "hotel-phone",
    "hotel-parking",
    "hotel-type",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-area",
    "attraction-price",
    "attraction-type",
    "attraction-address",
    "attraction-postcode",
    "attraction-phone",
    "attraction-area",
    "hospital-address",
    "hospital-postcode",
    "hospital-phone",
    "police-address",
    "police-postcode",
    "police-phone"
]

info_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "train": ["people", "leave", "destination", "day", "arrive", "departure"],
    "restaurant": ["time", "day", "people", "food", "pricerange", "name", "area"],
    "hotel": ["stay", "day", "people", "name", "area", "parking", "pricerange", "stars", "internet", "type"],
    "attraction": ["type", "name", "area"],
    "hospital": ["department"]
}

all_info_slots = [
    "taxi-leave",
    "taxi-destination",
    "taxi-departure",
    "taxi-arrive",
    "train-people",
    "train-leave",
    "train-destination",
    "train-day",
    "train-arrive",
    "train-departure",
    "restaurant-time",
    "restaurant-day",
    "restaurant-people",
    "restaurant-food",
    "restaurant-pricerange",
    "restaurant-name",
    "restaurant-area",
    "hotel-stay",
    "hotel-day",
    "hotel-people",
    "hotel-name",
    "hotel-area",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-internet",
    "hotel-type",
    "attraction-type",
    "attraction-name",
    "attraction-area",
    "hospital-department"
]

belief_state_mapping = {
    "taxi": {
        "leaveAt": "taxi-leave",
        "destination": "taxi-destination",
        "departure": "taxi-departure",
        "arriveBy": "taxi-arrive"
    },
    "train": {
        "leaveAt": "train-leave",
        "destination": "train-destination",
        "day": "train-day",
        "arriveBy": "train-arrive",
        "departure": "train-departure"
    },
    "restaurant": {
        "food": "restaurant-food",
        "pricerange": "restaurant-pricerange",
        "name": "restaurant-name",
        "area": "restaurant-area"
    },
    "hotel": {
        "name": "hotel-name",
        "area": "hotel-area",
        "parking": "hotel-parking",
        "pricerange": "hotel-pricerange",
        "stars": "hotel-stars",
        "internet": "hotel-internet",
        "type": "hotel-type"
    },
    "attraction": {
        "type": "attraction-type",
        "name": "attraction-name",
        "area": "attraction-area"
    },
    "hospital": {
        "department": "hospital-department"
    }
}

belief_state_mapping_reversed = {
    "taxi": {
        "taxi-leave": "leaveAt",
        "taxi-destination": "destination",
        "taxi-departure": "departure",
        "taxi-arrive": "arriveBy"
    },
    "train": {
        "train-leave": "leaveAt",
        "train-destination": "destination",
        "train-day": "day",
        "train-arrive": "arriveBy",
        "train-departure": "departure"
    },
    "restaurant": {
        "restaurant-food": "food",
        "restaurant-pricerange": "pricerange",
        "restaurant-name": "name",
        "restaurant-area": "area"
    },
    "hotel": {
        "hotel-name": "name",
        "hotel-area": "area",
        "hotel-parking": "parking",
        "hotel-pricerange": "pricerange",
        "hotel-stars": "stars",
        "hotel-internet": "internet",
        "hotel-type": "type"
    },
    "attraction": {
        "attraction-type": "type",
        "attraction-name": "name",
        "attraction-area": "area"
    },
    "hospital": {
        "hospital-department": "department"
    }
}

entry_slots = {
    "train": ["leave", "destination", "day", "arrive", "departure"],
    "restaurant": ["food", "pricerange", "name", "area"],
    "hotel": ["name", "area", "parking", "pricerange", "stars", "internet", "type"],
    "attraction": ["type", "name", "area"],
    "hospital": ["department"]
}

delex_slots = {
    "taxi": ["car", "phone"],
    "train": ["duration", "leave", "arrive", "id", "reference", "price", "choice"],
    "restaurant": ["postcode", "address", "food", "reference", "name", "phone", "pricerange", "area", "choice"],
    "hotel": ["address", "postcode", "type", "stars", "reference", "name", "phone", "pricerange", "area", "choice"],
    "attraction": ["type", "address", "postcode", "name", "phone", "price", "area", "choice"],
    "hospital": ["department", "phone", "postcode", "address", "name"],
    "police": ["name", "address", "phone", "postcode"]
}
# information not in DB
# hospital_name: "Addenbrookes Hospital"
# hospital_address: "Hills Rd, Cambridge"
# hospital_postcode: "CB20QQ"
# police_phone: "CB11JG"

dialog_acts = {
    "attraction": ["inform", "nooffer", "recommend", "request", "select"],
    "booking": ["book", "inform", "nobook", "request"],
    "hotel": ["inform", "nooffer", "recommend", "request", "select"],
    "restaurant": ["inform", "nooffer", "recommend", "request", "select"],
    "taxi": ["inform", "request"],
    "train": ["inform", "nooffer", "offerbook", "offerbooked", "request", "select"],
    "hospital": ["inform", "request"],
    "police": ["inform"],
    "general": ["bye", "greet", "reqmore", "welcome"]
}

dialogue_acts_slots = {
    "attraction-inform": ["area", "type", "choice", "postcode", "name", "phone", "address", "price", "open"],
    "attraction-nooffer": ["area", "type", "name", "choice", "address", "price"],
    "attraction-recommend": ["postcode", "name", "area", "phone", "address", "type", "choice", "price", "open"],
    "attraction-request": ["area", "type", "price", "name"],
    "attraction-select": ["type", "area", "name", "choice", "phone", "price", "address"],
    "booking-book": ["reference", "name", "people", "time", "day", "stay"],
    "booking-inform": ["stay", "name", "day", "people", "reference", "time"],
    "booking-nobook": ["reference", "stay", "day", "people", "time", "name"],
    "booking-request": ["time", "day", "people", "stay"],
    "general-bye": [],
    "general-greet": [],
    "general-reqmore": [],
    "general-welcome": [],
    "hotel-inform": ["name", "reference", "type", "choice", "address", "postcode", "area", "internet", "parking", "stars", "pricerange", "phone"],
    "hotel-nooffer": ["type", "pricerange", "area", "stars", "internet", "parking", "name", "choice"],
    "hotel-recommend": ["name", "pricerange", "area", "stars", "parking", "internet", "address", "type", "phone", "postcode", "choice"],
    "hotel-request": ["area", "pricerange", "stars", "type", "parking", "internet", "name"],
    "hotel-select": ["pricerange", "name", "choice", "type", "area", "stars", "parking", "internet", "address", "phone"],
    "restaurant-inform": ["postcode", "food", "name", "pricerange", "address", "phone", "area", "choice", "reference"],
    "restaurant-nooffer": ["food", "area", "pricerange", "choice", "name"],
    "restaurant-recommend": ["area", "pricerange", "name", "food", "address", "phone", "choice", "postcode"],
    "restaurant-request": ["area", "name", "pricerange", "food"],
    "restaurant-select": ["food", "area", "pricerange", "name", "choice", "address"],
    "taxi-inform": ["phone", "car", "departure", "destination", "leave", "arrive"],
    "taxi-request": ["destination", "leave", "departure", "arrive"],
    "train-inform": ["arrive", "id", "leave", "duration", "destination", "price", "departure", "day", "choice", "reference", "people"],
    "train-nooffer": ["departure", "destination", "leave", "day", "arrive", "id", "choice"],
    "train-offerbook": ["leave", "id", "people", "arrive", "destination", "departure", "day", "duration", "price", "choice", "reference"],
    "train-offerbooked": ["reference", "price", "people", "id", "leave", "departure", "destination", "duration", "arrive", "day", "choice"],
    "train-request": ["leave", "day", "departure", "destination", "arrive", "people"],
    "train-select": ["arrive", "leave", "id", "day", "price", "choice", "departure", "destination", "people"],
    "hospital-inform": ["address", "department", "phone", "postcode"],
    "hospital-request": ["department"],
    "police-inform": ["address", "name", "phone", "postcode"]
}

gate_idx = {
    "delete": 0,
    "update": 1,
    "dontcare": 2,
    "copy": 3
}

name_typo = {
    "the peking": "peking retaurant",
    "the gonvile hotel": "gonville hotel",
    "saint johns college": "saint john's college",
    "yipee noodle bar": "yippee noodle bar",
    "pizza hut fenditton": "pizza hut fen ditton",
    "queens college": "queens' college",
    "ABC Theatre": "abc theatre",
    "churchills college": "churchill college",
    "saint johns college": "saint john's college",
    "el shaddia guesthouse": "el shaddai",
    "college": "not mentioned",
    "cambridge": "not mentioned",
    "kings college": "king's college",
    "cafe uno": "caffe uno",
    "alpha milton guest house": "alpha-milton guest house",
    "NOT(hamilton lodge)": "hamilton lodge",
    "rosas bed and breakfast": "rosa's bed and breakfast"
}

important_actions = [
    "[hotel]-[inform]-name",
    "[hotel]-[inform]-phone",
    "[hotel]-[inform]-postcode",
    "[hotel]-[inform]-address",
    "[hotel]-[inform]-reference",
    "[restaurant]-[inform]-name",
    "[restaurant]-[inform]-phone",
    "[restaurant]-[inform]-postcode",
    "[restaurant]-[inform]-address",
    "[restaurant]-[inform]-reference",
    "[attraction]-[inform]-name",
    "[attraction]-[inform]-phone",
    "[attraction]-[inform]-postcode",
    "[attraction]-[inform]-address",
    "[train]-[inform]-id",
    "[train]-[inform]-reference",
    "[taxi]-[inform]-phone",
    "[booking]-[book]-reference",
    "[booking]-[book]-name",
    "[booking]-[inform]-name",
    "[train]-[offerbooked]-reference",
    "[train]-[offerbooked]-id",
    "[train]-[offerbook]-id"
]
next_response = {
    "description1" : "What does System will say next?"
}

QA = {
    "general-question" : "are they talking about",
    
    "hotel-pricerange": {
        "description1": "what is the price budget for the hotel?",
        "description2": "what is the cost of the hotel?",
        "values": [
            "cheap",
            "moderate",
            "expensive",
            "dontcare",
            "not mentioned"
        ]
    },
    
    "hotel-type": {
        "description1": "what is the type of hotel?",
        "description2": "what is the hotel type?",
        "values": [
            "hotel",
            "guesthouse",
            "dontcare",
            "not mentioned"
        ]
    },
    "hotel-parking": {
        "description1": "does the hotel have a parking lot?",
        "description2": "does the hotel have a parking lot?",
        "values": [
            "yes",
            "no",
            "free",
            "dontcare",
            "not mentioned"
        ]
    },
    "hotel-stay": {
        "description1": "how long did user book the hotel?",
        "description2": "how long did user book the hotel?"
    },
    "hotel-day": {
        "description1": "what day did user book the hotel?",
        "description2": "what day will user stay at the hotel?",
        "values": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "dontcare",
            "not mentioned"
        ]
    },
    "hotel-people": {
        "description1": "what is the number of people for the hotel booking?",
        "description2": "how many people will stay at the hotel?"
        
    },
    "hotel-area": {
        "description1": "where is the area of the hotel?",
        "description2": "where is the area of the hotel?"
        
    },
    "hotel-stars": {
        "description1": "what is the star rating of the hotel?",
        "description2": "what is the star rating of the hotel?",
        "values": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "dontcare",
            "not mentioned"
        ]
    },
    "hotel-internet": {
        "description1": "does the user want a hotel with internet access?",
        "description2": "does the user want a hotel with internet access?(multiple choice)",
        "values": [
            "yes",
            "no",
            "dontcare",
            "not mentioned"
        ]
    },
    
    "train-destination": {
        "description1": "where is destination of train?",
        "description2": "where is the train's destination?"
    },
    
    "train-day": {
        "description1": "what day did user book the train?",
        "description2": "what day did user book the train?",
        "values": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "dontcare",
            "not mentioned"
        ]
    },
    "train-departure": {
        "description1": "where is the departure location of train?",
        "description2": "where is the departure location of train?",
        
    },
    "train-arrive": {
        "description1": "when is the train arriving?",
        "description2": "when is the train arriving?",
        
    },
    "train-people": {
        "description1": "what is the number of people for the train booking?",
        "description2": "what is the number of people for the train booking?"
        
    },
    "train-leave": {
        "description1": "when is the train leaving?",
        "description2": "when is the train leaving?",
        
    },
    "attraction-area": {
        "description1": "what is the area of the attraction?",
        "description2": "where is the attraction?"
    
    },
    "restaurant-food": {
        "description1": "what kind of food do restaurants serve?",
        "description2": "what kind of food do restaurants serve?",
        
    },
    "restaurant-pricerange": {
        "description1": "what is the price budget for the restaurant?",
        "description2": "what is the price budget for the restaurant?",
        "values": [
            "expensive",
            "cheap",
            "moderate",
            "dontcare",
            "not mentioned"
        ]
    },
    "restaurant-area": {
        "description1": "where is the area of the restaurant?",
        "description2": "Where is the restaurant located?",
    },
    "attraction-name": {
        "description1": "what is attraction name?",
        "description2": "what is attraction name?",
        
    },
    "restaurant-name": {
        "description1": "what is restaurant name?",
        "description2": "what is the name of the restaurant?"
        
    },
    "attraction-type": {
        "description1": "what is the type of attraction?",
        "description2": "what are the types of attraction?"  
    },
    
    "hospital-department": {
        "description1": "what is department of hospital",
        "description2": "what is department of hospital"
    },
    "hotel-name": {
        "description1": "what is hotel name?",
        "description2": "what name hotel do users want to stay in?"
    },
    "taxi-leave": {
        "description1": "when is the taxi leaving?",
        "description2": "What time does the taxi leave?",
        
        
    },
    "taxi-destination": {
        "description1": "where is destination of taxi?",
        "description2": "where is destination of taxi?",
        
    },
    "taxi-departure": {
        "description1": "where is the departure location of taxi?",
        "description2": "where is the taxi's departure point?"
    },
    "restaurant-time": {
        "description1": "when is the book time of restaurant?",
        "description2": "when is the book time of restaurant?"
        
    },
    "restaurant-day": {
        "description1": "What day did user book the restaurant?",
        "description2": "when is the restaurant reservation day?",
        
        "values": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "dontcare",
            "not mentioned"
        ]
    },
    "restaurant-people": {
        "description1": "what is the number of people for the restaurant booking?",
        "description2": "how many people make a reservation for a restaurant?"
        
    },
    "taxi-arrive": {
        "description1": "when is the taxi arriving?",
        "description2": "when will the taxi arrive?",
    },
    "bus-departure": {
        "description1": "where is the departure location of bus?",
        "description2": "where is the departure location of bus?(extract)"
    },
    "bus-destination": {
        "description1": "where is destination of bus?",
        "description2": "where is destination of bus?(extract)",
        

    },
    "bus-leave": {
        "description1": "when is the bus leaving?",
        "description2": "when is the bus leaving?"
    },
    "bus-day": {
        "description1": "what day did user book the bus?",
        "description2": "what day did user book the bus?(multiple-choice)",
        "values": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "dontcare",
            "not mentioned"
        ]
    },
    "multichoice-domain" : [
        "hotel-pricerange",
        "hotel-type",
        "hotel-parking",
        "hotel-day", 
        "hotel-stars",
        "hotel-internet",
        "train-day",
        "restaurant-pricerange",
        "restaurant-day"
    ],
    "extract-domain" : [
        "attraction-type",
        "hotel-people",
        "hotel-area",
        "restaurant-area",
        "hotel-stay",
        "train-destination",
        "train-departure",
        "train-arrive",
        "train-people",
        "train-leave",
        "attraction-area",
        "restaurant-food",
        "attraction-name",
        "restaurant-name",
        "restaurant-people",
        "taxi-arrive",
        "hotel-name",
        "taxi-leave",
        "taxi-destination",
        "taxi-departure",
        "restaurant-time"
    ],
    
    "all-domain" : [
        "hotel-pricerange",
        "hotel-type",
        "hotel-parking",
        "hotel-day", 
        "hotel-stars",
        "hotel-internet",
        "train-day",
        "restaurant-pricerange",
        "restaurant-day",
        "attraction-type",
        "hotel-people",
        "hotel-area",
        "restaurant-area",
        "hotel-stay",
        "train-destination",
        "train-departure",
        "train-arrive",
        "train-people",
        "train-leave",
        "attraction-area",
        "restaurant-food",
        "attraction-name",
        "restaurant-name",
        "restaurant-people",
        "taxi-arrive",
        "hotel-name",
        "taxi-leave",
        "taxi-destination",
        "taxi-departure",
        "restaurant-time",
    ],
    
    'bigger-domain' : [
        'train',
        'taxi',
        'hotel',
        'restaurant',
        'attraction'
    ],
    
    "NOT_MENTIONED" : 'not mentioned',


}





 # all-domian에 next response있어야함