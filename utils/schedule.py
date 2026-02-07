"""
Training Schedule Utilities
Temperature, simulation count, and game count schedules
"""


def create_temperature_schedule(temp_start: float, temp_end: float):
    """Create temperature schedule function.
    
    Note: Temperature is now only applied for first 8 moves in self-play.
    This schedule controls the temperature value used during those moves.
    """
    def get_temperature(iteration: int, total_iterations: int) -> float:
        # Use constant temperature=1.0 for exploration in first 8 moves
        # After 8 moves, greedy selection is used regardless of this value
        return 1.0
    return get_temperature


def create_simulation_schedule(min_sim: int, max_sim: int):
    """Create simulation count schedule function."""
    def get_num_simulations(iteration: int, total_iterations: int) -> int:
        progress = iteration / total_iterations
        if progress < 0.2:
            return min_sim
        elif progress < 0.5:
            return min_sim + int((max_sim - min_sim) * 0.3)
        elif progress < 0.8:
            return min_sim + int((max_sim - min_sim) * 0.6)
        else:
            return max_sim
    return get_num_simulations


def create_games_schedule(min_games: int, max_games: int):
    """Create game count schedule function."""
    def get_num_games(iteration: int, total_iterations: int) -> int:
        progress = iteration / total_iterations
        if progress < 0.2:
            return min_games
        elif progress < 0.5:
            return min_games + int((max_games - min_games) * 0.3)
        elif progress < 0.8:
            return min_games + int((max_games - min_games) * 0.6)
        else:
            return max_games
    return get_num_games
