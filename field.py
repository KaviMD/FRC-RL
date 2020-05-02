import enum

class FieldCornerType(enum.Enum):
    normal = 1
    bevel = 2

class Body:
    def __init__(self, name, body):
        self.name = name
        self.body = body

class Obstacle(Body):
    def __init__(self, name, body):
        self.name = name
        self.body = body


class Zone(Body):
    def __init__(self, name, body, zone_id=0):
        self.name = name
        self.body = body
        self.zone_id = zone_id
    
    def contains_robot(self, robot):
        return False

class GameElement:
    def __init__(self, location, in_robot):
        self.in_robot = in_robot
        self.location = location

class Field():
    # Field Dimensions: 27 ft by 54 ft
    def __init__(self, width=54, height=27, corner_type=FieldCornerType.bevel):
        self.width = width
        self.height = height
        self.corner_type = corner_type

        self.field_boundary_obstacles = []
        self.generate_walls()

        self.field_element_obstacles = []

        self.robot_starting_location = Body("starting-location", "start")

        self.game_element_starting_locations = []

        self.zones = []

    def generate_walls(self):
        """
        Generate boundary walls to stop the robot from leaving the field
        """
        self.field_boundary_obstacles.append(Obstacle("upper-wall", "top"))
        self.field_boundary_obstacles.append(Obstacle("lower-wall", "bottom"))
        self.field_boundary_obstacles.append(Obstacle("left-wall", "left"))
        self.field_boundary_obstacles.append(Obstacle("right-wall", "right"))

        if self.corner_type == FieldCornerType.bevel:
            self.field_boundary_obstacles.append(Obstacle("upper-right-corner", "upper-right"))
            self.field_boundary_obstacles.append(Obstacle("upper-left-corner", "upper-left"))
            self.field_boundary_obstacles.append(Obstacle("lower-right-corner", "lower-right"))
            self.field_boundary_obstacles.append(Obstacle("lower-left-corner", "lower-left"))
    
    def add_obstacle(self, o: Obstacle):
        """
        Add a collidable object such as a switch (2018), or cargo ship (2019)
        """
        self.field_element_obstacles.append(o)
    
    def set_robot_starting_location(self, z: Zone):
        """
        Set an "spawn area" for the robots. The will all start the match somewhere in this area
        """
        self.robot_starting_location = z
    
    def add_game_element_starting_location(self, location, element):
        """
        Add a game element starting location to the field.
        
        Arguments:
            - location: (x,y) ft
            - element: GameElement
        """
        self.game_element_starting_locations.append(location)
    
    def create_zone(self, name, body):
        """
        Add a zone to the field
        """
        self.zones.append(Zone(name, body, len(self.zones)))
    
    def get_robot_zone(self, robot):
        for zone in self.zones:
            if zone.contains_robot(robot):
                return (zone.id, zone.name)