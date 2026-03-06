from dataclasses import dataclass
from typing import Callable

# hypothetical imports from langgraph; adjust names if library differs
try:
    from langgraph import Graph, Node
except ImportError:  # fallback stubs for linting/testing without the package
    class Node:
        def __init__(self, func: Callable, name: str = None):
            self.func = func
            self.name = name or func.__name__
            self.edges = []

        def run(self, state):
            return self.func(state)

    class Graph:
        def __init__(self):
            self.nodes = []
            self.connections = []

        def add_nodes(self, nodes):
            self.nodes.extend(nodes)

        def connect(self, src, dst, condition=None):
            self.connections.append((src, dst, condition))

        def run(self, state):
            # simple traversal: start at first node and follow connections
            current = self.nodes[0]
            while True:
                state = current.run(state)
                # find next
                next_node = None
                for src, dst, cond in self.connections:
                    if src is current and (cond is None or cond(state)):
                        next_node = dst
                        break
                if not next_node:
                    break
                current = next_node
            return state


from rag import get_relevant_docs
from tools import is_math_expression, safe_eval
from config import generate_response


@dataclass
class AgentState:
    question: str = ""
    context: str = ""
    answer: str = ""
    tool_used: str = ""


def retrieval_node(state: AgentState) -> AgentState:
    docs = get_relevant_docs(state.question)
    state.context = "\n\n".join(docs)
    return state


def decision_node(state: AgentState) -> AgentState:
    q = state.question or ""
    if is_math_expression(q):
        state.tool_used = "calculator"
    elif q.strip():
        # If there are digits or keywords, treat as RAG
        state.tool_used = "rag"
    else:
        state.tool_used = "direct"
    return state


def tool_node(state: AgentState) -> AgentState:
    if state.tool_used == "calculator":
        state.answer = safe_eval(state.question)
    return state


def generate_node(state: AgentState) -> AgentState:
    prompt = state.question
    if state.context:
        prompt += "\n\nContext:\n" + state.context
    state.answer = generate_response(prompt)
    return state


# build graph
graph = Graph()

node_decision = Node(decision_node, name="decision")
node_retrieval = Node(retrieval_node, name="retrieval")
node_tool = Node(tool_node, name="tool")
node_generate = Node(generate_node, name="generate")

# add nodes in logical order
graph.add_nodes([node_decision, node_retrieval, node_tool, node_generate])

# connect edges with simple conditions
graph.connect(node_decision, node_tool, condition=lambda s: s.tool_used == "calculator")
graph.connect(node_decision, node_retrieval, condition=lambda s: s.tool_used == "rag")
# direct jump to generate if nothing else
graph.connect(node_decision, node_generate, condition=lambda s: s.tool_used == "direct")

# after retrieval or tool we always go to generation (tool may overwrite answer but generate may still run)
graph.connect(node_retrieval, node_generate)
graph.connect(node_tool, node_generate)


def run_agent(question: str) -> AgentState:
    state = AgentState(question=question)
    final = graph.run(state)
    return final
