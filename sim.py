import field
from field import Obstacle, Zone

class PowerCell(field.GameElement):
    def __init__(self):
        super.__init__()

infinite_recharge_field = field.Field()
infinite_recharge_field.add_obstacle(Obstacle("blue-trench", "trench"))
infinite_recharge_field.set_robot_starting_location(Zone("initiation-line","starting-location"))
infinite_recharge_field.add_game_element_starting_location((0.5, 10), PowerCell)
infinite_recharge_field.add_game_element_starting_location((0.5, 12), PowerCell)
infinite_recharge_field.add_game_element_starting_location((0, 14), PowerCell)
infinite_recharge_field.add_game_element_starting_location((1, 14), PowerCell)
infinite_recharge_field.create_zone("blue-power-port","square")