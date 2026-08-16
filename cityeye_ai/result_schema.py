def create_result(video_name):
    return {
        "project": "CityEye",
        "video": video_name,

        "summary": {
            "total_vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "persons": 0
        },

        "events": {
            "helmet_violation": {
                "detected": False,
                "count": 0
            },

            "triple_riding": {
                "detected": False,
                "count": 0
            },

            "wrong_way": {
                "detected": False,
                "count": 0
            },

            "accident": {
                "detected": False,
                "count": 0
            }
        },

        "event_details": []
    }