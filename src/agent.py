import sys
import time

sys.path.insert(0, r"..\lib")

import MalmoPython

MISSION_FILE = r"..\missions\training\craft.xml"


def get_mission_xml():
    with open(MISSION_FILE, "r") as f:
        return f.read()


agent_host = MalmoPython.AgentHost()


client_pool = MalmoPython.ClientPool()
client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

print("Starting mission")

agent_host.startMission(MalmoPython.MissionSpec(get_mission_xml(), True), client_pool, MalmoPython.MissionRecordSpec(), 0, "test")

world_state = agent_host.getWorldState()

while not world_state.has_mission_begun:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()


print("Mission running")
