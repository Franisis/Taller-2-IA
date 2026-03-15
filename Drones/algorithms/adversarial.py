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

            
            # Caso base
            if estado.is_win() or estado.is_lose() or profundidad == 0:
                return self.evaluation_function(estado)
            
            num_agentes = estado.get_num_agents()
            acciones_legales = estado.get_legal_actions(agente)
            
            # Si no hay acciones evaluamos el estado actual
            if not acciones_legales:
                return self.evaluation_function(estado)
            
            siguiente_profundidad = profundidad - 1 if (agente + 1) % num_agentes == 0 else profundidad
            siguiente_agente = (agente + 1) % num_agentes
            
            if agente == 0:  # Es turno del DRON MAX
                valor_max = float('-inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = minimax(sucesor, siguiente_profundidad, siguiente_agente)
                    valor_max = max(valor_max, valor)  # Queremos el máximo
                return valor_max
            else:  # Es turno de los CAZADORES MIN
                valor_min = float('inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = minimax(sucesor, siguiente_profundidad, siguiente_agente)
                    valor_min = min(valor_min, valor)  # Queremos el mínimo 
                return valor_min
        
        # Buscamos la mejor acción para el dron en el nivel raíz
        acciones_legales = estado.get_legal_actions(self.index)
        if not acciones_legales:
            return None
        
        mejor_accion = None
        mejor_valor = float('-inf')
        num_agentes = estado.get_num_agents()
        
        for accion in acciones_legales:
            sucesor = estado.generate_successor(self.index, accion)
            # Bajamos profundidad después del turno del dron primero juega dron, luego cazadores
            siguiente_profundidad = self.depth - 1 if num_agentes == 1 else self.depth
            valor = minimax(sucesor, siguiente_profundidad, 1)  
            
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion
        
        return mejor_accion


class AlphaBetaAgent(MultiAgentSearchAgent):

    def get_action(self, state: GameState) -> Directions | None:

        
        def alpha_beta(estado: GameState, profundidad: int, agente: int, 
                      alfa: float, beta: float) -> float:

            
            # Caso base
            if estado.is_win() or estado.is_lose() or profundidad == 0:
                return self.evaluation_function(estado)
            
            num_agentes = estado.get_num_agents()
            acciones_legales = estado.get_legal_actions(agente)
            
            if not acciones_legales:
                return self.evaluation_function(estado)
            
            siguiente_profundidad = profundidad - 1 if (agente + 1) % num_agentes == 0 else profundidad
            siguiente_agente = (agente + 1) % num_agentes
            
            if agente == 0:  # Turno del DRON (MAX)
                valor_max = float('-inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = alpha_beta(sucesor, siguiente_profundidad, siguiente_agente, alfa, beta)
                    valor_max = max(valor_max, valor)
                    alfa = max(alfa, valor)
                
                return valor_max
            else:  # Turno de los CAZADORES (MIN)
                valor_min = float('inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = alpha_beta(sucesor, siguiente_profundidad, siguiente_agente, alfa, beta)
                    valor_min = min(valor_min, valor)
                    beta = min(beta, valor)
                
                return valor_min
        
        # Encontramos la mejor acción en la raíz
        acciones_legales = estado.get_legal_actions(self.index)
        if not acciones_legales:
            return None
        
        mejor_accion = None
        mejor_valor = float('-inf')
        num_agentes = estado.get_num_agents()
        alfa = float('-inf')  # El dron busca lo máximo
        beta = float('inf')   # Los cazadores buscan lo mínimo
        
        for accion in acciones_legales:
            sucesor = estado.generate_successor(self.index, accion)
            siguiente_profundidad = self.depth - 1 if num_agentes == 1 else self.depth
            valor = alpha_beta(sucesor, siguiente_profundidad, 1, alfa, beta)
            
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion
            alfa = max(alfa, valor)
        
        return mejor_accion


class ExpectimaxAgent(MultiAgentSearchAgent):


    def get_action(self, state: GameState) -> Directions | None:
        
        def expectimax(estado: GameState, profundidad: int, agente: int) -> float:
            
            # Caso base
            if estado.is_win() or estado.is_lose() or profundidad == 0:
                return self.evaluation_function(estado)
            
            num_agentes = estado.get_num_agents()
            acciones_legales = estado.get_legal_actions(agente)
            
            if not acciones_legales:
                return self.evaluation_function(estado)
            
            siguiente_profundidad = profundidad - 1 if (agente + 1) % num_agentes == 0 else profundidad
            siguiente_agente = (agente + 1) % num_agentes
            
            if agente == 0:  # Turno del DRON 
                valor_max = float('-inf')
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = expectimax(sucesor, siguiente_profundidad, siguiente_agente)
                    valor_max = max(valor_max, valor)  # Queremos el máximo
                return valor_max
            else:  # Turno de CAZADORES 
                # valores de todas las acciones posibles
                valores_hijos = []
                for accion in acciones_legales:
                    sucesor = estado.generate_successor(agente, accion)
                    valor = expectimax(sucesor, siguiente_profundidad, siguiente_agente)
                    valores_hijos.append(valor)
                
          
                # valor_final = (1-p) * peor_caso + p * promedio
                valor_minimo = min(valores_hijos)  # Si juega óptimo 
                valor_promedio = sum(valores_hijos) / len(valores_hijos)  # Si elige aleatorio
                
                # Combinamos según la probabilidad
                valor_mixto = (1 - self.prob) * valor_minimo + self.prob * valor_promedio
                return valor_mixto
        
        # Encontramos la mejor acción para el dron
        acciones_legales = estado.get_legal_actions(self.index)
        if not acciones_legales:
            return None
        
        mejor_accion = None
        mejor_valor = float('-inf')
        num_agentes = estado.get_num_agents()
        
        for accion in acciones_legales:
            sucesor = estado.generate_successor(self.index, accion)
            siguiente_profundidad = self.depth - 1 if num_agentes == 1 else self.depth
            valor = expectimax(sucesor, siguiente_profundidad, 1)  
            
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion
        
        return mejor_accion
