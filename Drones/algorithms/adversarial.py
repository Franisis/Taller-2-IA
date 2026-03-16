from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # 0=greedy, 1=random
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:

        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):

    def get_action(self, state: GameState) -> Directions | None:

        def minimax(estado: GameState, profundidad: int, agente: int) -> float:
            if estado.is_win() or estado.is_lose() or profundidad == 0:
                return self.evaluation_function(estado)

            num_agentes = estado.get_num_agents()
            acciones_legales = estado.get_legal_actions(agente)

            if not acciones_legales:
                return self.evaluation_function(estado)

            siguiente_agente = (agente + 1) % num_agentes
            siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

            if agente == 0:  # MAX (dron)
                return max(
                    minimax(estado.generate_successor(agente, a), siguiente_profundidad, siguiente_agente)
                    for a in acciones_legales
                )
            else:  # MIN (cazadores)
                return min(
                    minimax(estado.generate_successor(agente, a), siguiente_profundidad, siguiente_agente)
                    for a in acciones_legales
                )

        # Raíz: elegir la mejor acción para el dron
        acciones_legales = state.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        num_agentes = state.get_num_agents()
        siguiente_agente = 1 % num_agentes  # Agente que sigue al dron
        # Si solo hay 1 agente (el dron), se reduce profundidad desde ya
        siguiente_profundidad = self.depth - 1 if siguiente_agente == 0 else self.depth

        mejor_accion = max(
            acciones_legales,
            key=lambda a: minimax(
                state.generate_successor(self.index, a),
                siguiente_profundidad,
                siguiente_agente
            )
        )
        return mejor_accion


class AlphaBetaAgent(MultiAgentSearchAgent):

    def get_action(self, state: GameState) -> Directions | None:

        def alpha_beta(estado: GameState, profundidad: int, agente: int,
                       alfa: float, beta: float) -> float:
            if estado.is_win() or estado.is_lose() or profundidad == 0:
                return self.evaluation_function(estado)

            num_agentes = estado.get_num_agents()
            acciones_legales = estado.get_legal_actions(agente)

            if not acciones_legales:
                return self.evaluation_function(estado)

            siguiente_agente = (agente + 1) % num_agentes
            siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

            if agente == 0:  # MAX (dron)
                valor = float('-inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = max(valor, alpha_beta(sucesor, siguiente_profundidad, siguiente_agente, alfa, beta))
                    alfa = max(alfa, valor)
                    if valor >= beta:  # Poda beta
                        break
                return valor
            else:  # MIN (cazadores)
                valor = float('inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = min(valor, alpha_beta(sucesor, siguiente_profundidad, siguiente_agente, alfa, beta))
                    beta = min(beta, valor)
                    if valor <= alfa:  # Poda alfa
                        break
                return valor

        acciones_legales = state.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        num_agentes = state.get_num_agents()
        siguiente_agente = 1 % num_agentes
        siguiente_profundidad = self.depth - 1 if siguiente_agente == 0 else self.depth

        mejor_accion = None
        mejor_valor = float('-inf')
        alfa = float('-inf')
        beta = float('inf')

        for accion in acciones_legales:
            sucesor = state.generate_successor(self.index, accion)
            valor = alpha_beta(sucesor, siguiente_profundidad, siguiente_agente, alfa, beta)
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion
            alfa = max(alfa, valor)

        return mejor_accion


class ExpectimaxAgent(MultiAgentSearchAgent):

    def get_action(self, state: GameState) -> Directions | None:

        def expectimax(estado: GameState, profundidad: int, agente: int) -> float:
            if estado.is_win() or estado.is_lose() or profundidad == 0:
                return self.evaluation_function(estado)

            num_agentes = estado.get_num_agents()
            acciones_legales = estado.get_legal_actions(agente)

            if not acciones_legales:
                return self.evaluation_function(estado)

            siguiente_agente = (agente + 1) % num_agentes
            siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

            if agente == 0:  # MAX (dron)
                return max(
                    expectimax(estado.generate_successor(agente, a), siguiente_profundidad, siguiente_agente)
                    for a in acciones_legales
                )
            else:  # Nodo de azar (cazadores)
                valores = [
                    expectimax(estado.generate_successor(agente, a), siguiente_profundidad, siguiente_agente)
                    for a in acciones_legales
                ]
                valor_minimo = min(valores)   # Comportamiento greedy (p=0)
                valor_promedio = sum(valores) / len(valores)  # Comportamiento aleatorio (p=1)
                return (1 - self.prob) * valor_minimo + self.prob * valor_promedio

        acciones_legales = state.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        num_agentes = state.get_num_agents()
        siguiente_agente = 1 % num_agentes
        siguiente_profundidad = self.depth - 1 if siguiente_agente == 0 else self.depth

        mejor_accion = max(
            acciones_legales,
            key=lambda a: expectimax(
                state.generate_successor(self.index, a),
                siguiente_profundidad,
                siguiente_agente
            )
        )
        return mejor_accion