# multi_agent_untangle_commit.py

from openai import OpenAI
import re
import json
import os
import time
import argparse
from typing import Dict, Any, List
from enum import Enum
import concurrent.futures
import tqdm
from evaluator import Evaluator
from utils.json_utils import save_json, load_json, preprocess_response_string
from utils.config_utils import Config
# Placeholder for the OpenAI LLM configuration

class LLMType(Enum):
    DEEPSEEK = {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    }

LLM_MODEL = {}

class AgentType(Enum):
    """Agent type enumeration."""
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    REVIEW = "review"



AgentDescriptionMap = {
    AgentType.EXPLICIT: "Analyzes explicit code dependencies and control dependencies.",
    AgentType.IMPLICIT: "Analyzes implicit relationships between changes based on context, including context similarity, clone patterns, refactoring and similar code blocks.",
    AgentType.REVIEW: "Synthesizes opinions from explicit and implicit agents."
}


class BaseAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = OpenAI(
            api_key=LLM_MODEL["api_key"],
            base_url=LLM_MODEL["base_url"]
        )
        self.model_name = LLM_MODEL["model_name"]
        self.memory = []

    def call_llm(self, system_message: str, user_message: str, max_retries: int = 3) -> str:
        retries = 0
        while retries < max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"} ,
                    extra_body={"enable_thinking": False},
                    stream=True
                )
                # Handle streaming response
                response_chunks = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)

                response = "".join(response_chunks)

                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(0.5)  # Brief pause before retrying


class ExplicitAgent(BaseAgent):
    def __init__(self, agent_id: str, agent_type: AgentType = AgentType.EXPLICIT, model_name: str = "deepseek-chat"):
        super().__init__(agent_id)
        self.agent_type = agent_type
        self.model_name = model_name
        print(f"Initialized ExplicitAgent with ID: {self.agent_id}, Type: {self.agent_type}, Model: {self.model_name}")


    def analyze(self, commit_diff: Dict) -> Dict[str, Any]:
        system_prompt = f"""
You are an expert in code dependency analysis, specializing in analyzing explicit dependencies between code changes and comprehend their underlying semantic relationships — including both data dependencies (such as variable or value flows) and control dependencies (such as conditional structures and execution order). 
Given: - A commit diff in the following format: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}}.  Task: - Group related code changes into groups, where changes in the same group are highly related via data or control dependencies.
- Ensure that changes in different groups are independent from each other. - Ignore file boundaries when grouping even if changes are in the same file, they can belong to different groups if they are not data/control-dependent. The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern.
Output format (JSON): ```json
{{
  "answer": {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},
  "explanation": "Explain your grouping decisions, highlighting key dependencies."
}}
        """

        user_message = f"Commit diff: {commit_diff} Please analyze the commit diff using the PDG and output the grouping in JSON format with 'answer' and 'explanation' fields. Remember: group changes based on data and control dependencies, not file paths."

        response_text = self.call_llm(system_message=system_prompt, user_message=user_message)
        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f'ExplicitAgent {self.agent_id} reponse successfully parsed')
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"ExplicitAgent  {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result

    def review(self, questions: str, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        print(f'ExplicitAgent {self.agent_id} reviewing synthesis result with model {self.model_name}')# Get current round
        current_round = len(self.memory) // 2 + 1

        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break
         # Prepare system message for review
        system_message = {
            "role": "system",
"content": f"You are an expert in code dependency analysis, participating in round {current_round} of a multidisciplinary team consultation. You specialize in analyzing explicit dependencies between code changes and comprehend their underlying semantic relationships, including both data dependencies (e.g., variable or value flows) and control dependencies (e.g., conditional structures and execution order). "
f"Review the synthesis of multiple experts' opinions and determine if you agree with the conclusion. The commit diff format is: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}} The previous untangled result format is: {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}}. "
f"Consider your previous analysis and the ReviewAgent's synthesized opinion to decide whether to agree or provide a different perspective. "
f"The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern."
f"Output your response strictly in JSON format with the following fields: 'agree' (boolean or 'yes'/'no'), 'reason' (why you agree or disagree), and 'answer' (alternative answer if you disagree, otherwise empty)."
        }


        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nAnswer: {own_analysis.get('answer', '')}\n\n"

        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested answer: {synthesis.get('answer', '')}"


        user_message = {
            "type": "text",
            "text": f"Original question: {questions}\n\n"
                  f"{own_analysis_text}"
                  f"{synthesis_text}\n\n"
                  f"Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
                  f"1. 'agree': 'yes'/'no'\n"
                  f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                  f"3. 'answer': If you disagree, provide your own grouping in the format {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}}; otherwise, provide an empty object {{}}."
        }
        response_text = self.call_llm(system_message=system_message['content'], user_message=user_message['text'])

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f'ExplicitAgent {self.agent_id} reponse successfully parsed')

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"ExplicitAgent {self.agent_id} response is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}

            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line

            # Ensure required fields
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "answer" not in result:
                # Default to own previous answer or synthesized answer
                if own_analysis and "answer" in own_analysis:
                    result["answer"] = own_analysis["answer"]
                else:
                    result["answer"] = synthesis.get("answer", "No answer provided")

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result

class ImplicitAgent(BaseAgent):
    def __init__(self, agent_id: str, agent_type: AgentType = AgentType.IMPLICIT, model_name: str = "deepseek-chat"):
        super().__init__(agent_id)
        self.agent_type = agent_type
        self.model_name = model_name
        print(f"Initialized ImplicitAgent with ID: {self.agent_id}, Type: {self.agent_type}, Model: {self.model_name}")

    def analyze(self, commit_diff: Dict) -> Dict[str, Any]:

        system_prompt = f"""
You are an expert in code dependency analysis. Your task is to analyze implicit dependencies between code changes and comprehend their underlying semantic relationships, including contextual relationships, naming and structural similarities, code clones, and refactorings that might implicitly connect changes even if explicit data or control dependencies are not evident.
Given: - A commit diff in the following format: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}}.  Task: - Group related code changes into groups, where changes in the same group are highly related. - Ensure that changes in different groups are independent from each other. - Do not group changes solely by file paths; prioritize logical purpose, context, and relationships. The following principles are provided to guide your analysis: - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern.
Output format (JSON): ```json
{{
  "answer": {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},
  "explanation": "Explain your grouping decisions, highlighting key dependencies."
}}
        """

        user_message = f"Commit diff: {commit_diff} Provide your analysis in JSON format, including 'explanation' and 'answer' fields. Group changes based on their purpose and implicit relationships (naming, structure, refactoring), rather than just file paths."
        response_text = self.call_llm(system_message=system_prompt, user_message=user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f'ImplicitAgent {self.agent_id} reponse successfully parsed')
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"ImplicitAgent {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
        

    def review(self, questions: str, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        print(f'ImplicitAgent {self.agent_id} reviewing synthesis result with model {self.model_name}')
        # Get current round
        current_round = len(self.memory) // 2 + 1
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break
        # Prepare system message for review
        system_message = {
            "role": "system",
            "content": f"You are an expert in structural code analysis, participating in round {current_round} of a multidisciplinary team consultation. You specialize in analyzing implicit dependencies between code changes and comprehend their underlying semantic relationships, includeing identifying contextual relationships, code similarity, naming similarity, code clones, and refactorings that might implicitly connect changes even when explicit data or control dependencies are not evident.   "
                      f"Review the synthesis of multiple experts' opinions and determine if you agree with the conclusion. The commit diff format is: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}} The previous untangled result format is: {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}}."
                      f"Consider your previous analysis and the ReviewAgent's synthesized opinion to decide whether to agree or provide a different perspective. "
                      f"The following principles are provided to guide your analysis: - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern."
                      f"Output your response strictly in JSON format with the following fields: 'agree' (boolean or 'yes'/'no'), 'reason' (why you agree or disagree), and 'answer' (alternative answer if you disagree, otherwise empty)."
        }

        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nAnswer: {own_analysis.get('answer', '')}\n\n"
        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested answer: {synthesis.get('answer', '')}"
        user_content = {
            "type": "text",
            "text": f"Original question: {questions}\n\n"
                  f"{own_analysis_text}"
                  f"{synthesis_text}\n\n"
                  f"Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
                  f"1. 'agree': 'yes'/'no'\n"
                  f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                  f"3. 'answer': If you disagree, provide your own grouping in the format {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}}; otherwise, provide an empty object {{}}."
        }

        response_text = self.call_llm(system_message=system_message['content'], user_message=user_content['text'])
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f'ImplicitAgent {self.agent_id} reponse successfully parsed')

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"ImplicitAgent {self.agent_id} response is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}

            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line

            # Ensure required fields
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "answer" not in result:
                # Default to own previous answer or synthesized answer
                if own_analysis and "answer" in own_analysis:
                    result["answer"] = own_analysis["answer"]
                else:
                    result["answer"] = synthesis.get("answer", "No answer provided")

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        


class ReviewAgent(BaseAgent):
    def __init__(self, agent_id: str, agent_type: AgentType = AgentType.REVIEW, model_name: str = "deepseek-chat"):
        super().__init__(agent_id)
        self.agent_type = agent_type
        self.model_name = model_name
        print(f"Initialized ReviewAgent with ID: {self.agent_id}, Type: {self.agent_type}, Model: {self.model_name}")

    def synthesize(self, commit_diff: str, opinions: List[Dict[str, Any]], specialties: List[AgentType], current_round: int) -> Dict[str, Any]:
        print(f'ReviewAgent synthesizing round {current_round} opinions with model {self.model_name}')
        system_prompt = {
            "role": "system",
            "content": f"You are a review coordinator facilitating round {current_round} of a multidisciplinary team consultation. Your task is to synthesize the opinions of the explicit and implicit agents, taking into account their specialties and previous analyses."
                        "Explicit agents focus on explicit dependencies (e.g., data dependencies, control dependencies), while implicit agents focus on implicit dependencies (e.g., naming similarity, code similarity, code clones, refactorings, contextual relationships). Weigh both perspectives carefully to provide a balanced and coherent grouping. Groupings should be based on code dependencies rather than file boundaries. The commit diff format is: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}} The previous untangled result format is: {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}}. "
                        "The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern. - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern."
                        "Output format (JSON): ```json {'answer': {filename: [(symbol (- or +), linenumber, groupnumber), ...]},'explanation': 'Explanation of the grouping'}"
        }

        # Format doctors' opinions as input
        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(opinions, specialties)):
            formatted_opinion = f"Agent {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Agent Function: {AgentDescriptionMap[specialty]}\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Answer: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)
        opinions_text = "\n".join(formatted_opinions)

        # prepare current round's question
        questions = f"Commit diff: {commit_diff} Please untangle the commit diff."

        # Prepare user message with all opinions
        user_message = {
            "role": "user",
            "content": f"Question: {questions}\n\n"
                      f"Round {current_round} Agents' Opinions:\n{opinions_text}\n\n"
                      f"Please synthesize these opinions into a single consensus view, taking into account the specialties and perspectives of each agent. Output format (JSON): ```json {{'answer': {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},'explanation': 'Explanation of the grouping'}}"
        }
        # Call LLM with retry mechanism
        response_text = self.call_llm(system_prompt['content'], user_message['content'])
        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f'ReviewAgent {self.agent_id} reponse successfully parsed')

            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"ReviewAgent {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result


    def make_final_decision(self, commit_diff: str, reviews: List[Dict[str, Any]], specialties: List[AgentType], current_synthesis: Dict[str, Any], current_round: int, max_rounds: int) -> Dict[str, Any]:
        print(f'ReviewAgent making final decision for round {current_round} with model {self.model_name}')
        all_agree = all(review.get('agree', False) for review in reviews)
        reached_max_rounds = current_round >= max_rounds

        # Prepare system message for final decision
        system_message = f"You are a review coordinator facilitating round {current_round} of a multidisciplinary team consultation. The team has reached the maximum number of rounds without reaching full consensus. As the review coordinator, your task is to synthesize the final recommendation by applying a majority opinion approach, carefully weighing each agent's perspective on code structure dependencies. Explicit dependencies include data flows and control structures; implicit dependencies include naming similarity, code clones, refactorings, and contextual relationships. The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern. - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern."

        system_message += (
            "The commit diff format is: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}} The previous untangled result format is: {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}}. "
        )

        system_message += (
            "Output format: ```json {'answer': {filename: [(symbol (- or +), linenumber, groupnumber), ...]},'explanation': 'final reasoning'}"
        )
        # Format doctor reviews
        formatted_reviews = []
        for i, (review, specialty) in enumerate(zip(reviews, specialties)):
            formatted_review = f"Agent {i+1} ({specialty.value}):\n"
            formatted_review += f"Agree: {'Yes' if review.get('agree', False) else 'No'}\n"
            formatted_review += f"Reason: {review.get('reason', '')}\n"
            if not review.get('agree', False):
                formatted_review += f"Answer: {review.get('answer', '')}\n"
            formatted_reviews.append(formatted_review)

        reviews_text = "\n".join(formatted_reviews)
        # Prepare current synthesis text
        current_synthesis_text = (
            f"Current synthesized explanation: {current_synthesis.get('explanation', '')}\n"
            f"Current suggested answer: {current_synthesis.get('answer', '')}"
        )

        decision_type = "final" if all_agree or reached_max_rounds else "current round"

        # Review previous rounds' syntheses from memory
        previous_syntheses = []
        for i, mem in enumerate(self.memory):
            if mem["type"] == "synthesis" and mem["round"] < current_round:
                prev = f"Round {mem['round']} synthesis:\n"
                prev += f"Explanation: {mem['content'].get('explanation', '')}\n"
                prev += f"Answer: {mem['content'].get('answer', '')}"
                previous_syntheses.append(prev)

        previous_syntheses_text = "\n\n".join(previous_syntheses) if previous_syntheses else "No previous syntheses available."
        questions = f"Commit diff: {commit_diff} Please untangle the commit diff."
        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"Question: {questions}\n\n"
                      f"{current_synthesis_text}\n\n"
                      f"Agents' Reviews:\n{reviews_text}\n\n"
                      f"Previous Syntheses:\n{previous_syntheses_text}\n\n"
                      f"Please provide your {decision_type} decision, "
                      f"in JSON format, including 'explanation' and 'answer' fields."
        }
        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message['content'])
        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f'ReviewAgent {self.agent_id} reponse successfully parsed')
            # Add to memory
            self.memory.append({
                "type": "decision",
                "round": current_round,
                "final": all_agree or reached_max_rounds,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"ReviewAgent {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "decision",
                "round": current_round,
                "final": all_agree or reached_max_rounds,
                "content": result
            })
            return result

class CommitUntangleCoordinator:
    def __init__(self,
                max_rounds: int = 3,
                agent_configs: List[Dict] = None,
                meta_model_key: str = "deepseek-chat"):
        self.max_rounds = max_rounds
        self.agent_configs = agent_configs or [
            {"agent_id": "explicit_agent", "agent_type": AgentType.EXPLICIT, "model_name": meta_model_key},
            {"agent_id": "implicit_agent", "agent_type": AgentType.IMPLICIT, "model_name": meta_model_key},
        ]
        self.meta_model_key = meta_model_key
        self.agents =[]
        for idx, config in enumerate(self.agent_configs):
            if config["agent_type"] == AgentType.EXPLICIT:
                self.agents.append(ExplicitAgent(agent_id=config["agent_id"], agent_type=config["agent_type"], model_name=config["model_name"]))
            elif config["agent_type"] == AgentType.IMPLICIT:
                self.agents.append(ImplicitAgent(agent_id=config["agent_id"], agent_type=config["agent_type"], model_name=config["model_name"]))
        self.review_agent = ReviewAgent(agent_id="review_agent", agent_type=AgentType.REVIEW, model_name=meta_model_key)
        agent_info = ", ".join([
            f"{config['agent_type'].value} ({config.get('model_name', 'default')})"
            for config in self.agent_configs
        ])
        print(f"Initialized Coordinator, max_rounds={self.max_rounds}, agents: {agent_info}")

    def run(self, qid: str, commit_diff: str) -> Dict[str, Any]:
        start_time = time.time()

        print(f"Starting CUCDT consultation for case {qid}")

        # Case consultation history
        case_history = {
            "rounds": []
        }

        current_round = 0
        final_decision = None
        consensus_reached = False

        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": []}

            # Step 1: Each agent analyzes the commit diff
            print('Step 1: Agents analyzing commit diff')
            opinions = []
            for agent in self.agents:
                if isinstance(agent, ExplicitAgent):
                    opinion = agent.analyze(commit_diff)
                elif isinstance(agent, ImplicitAgent):
                    opinion = agent.analyze(commit_diff)
                ''' opinion format:
                {
                'answer': []
                'explanation': []
                }
                '''
                opinions.append(opinion)
                round_data["opinions"].append({
                    "agent_id": agent.agent_id,
                    "type": agent.agent_type.value,
                    "opinion": opinion
                })
            # Step 2: Meta agent synthesizes opinions
            print('Step 2: ReviewAgent synthesizing opinions')
            synthesis = self.review_agent.synthesize(
                commit_diff=commit_diff,
                opinions=opinions,
                specialties=[agent.agent_type for agent in self.agents],
                current_round=current_round
            )
            # synthesis ={}

            '''
            sysnthesis_format: {
                "answer": "Synthesized answer based on agents' opinions",
                "explanation": "Comprehensive reasoning for the synthesis"
            }
            '''
            round_data["synthesis"] = synthesis

            print(f"ReviewAgent synthesis: {synthesis.get('answer', '')}")

            # Step 3: Review synthesis
            print('Step 3: Agents reviewing synthesis')
            reviews = []
            all_agree = True
            for i, agent in enumerate(self.agents):
                print(f"Agent {i+1} ({agent.agent_type.value}) reviewing synthesis")
                questions = f"Commit diff: {commit_diff} Please untangle the commit diff."
                '''
                review_format: {
                    "agree": "yes"/"no",
                    "reason": "Rationale for agreeing or disagreeing",
                    "answer": "Your suggested answer if you disagree, or the synthesized answer if you agree"
                '''
                review = agent.review(questions=questions, synthesis=synthesis)
                reviews.append(review)
                round_data["reviews"].append({
                    "agent_id": agent.agent_id,
                    "type": agent.agent_type.value,
                    "review": review
                })
                agrees = review.get("agree", False)
                all_agree = all_agree and agrees
                print(f"Agent {i+1} agree: {'Yes' if agrees else 'No'}")

            # Add round data to history
            case_history["rounds"].append(round_data)         

            # Step 4: Final_Decision
            # consensus -> get answer  no_consensus: make_final_decision
            if all_agree:
                consensus_reached = True
                final_decision = synthesis
                print("Consensus reached")
            else:
                if current_round == self.max_rounds:
                    decision = self.review_agent.make_final_decision(
                        commit_diff=commit_diff,
                        reviews=reviews,
                        specialties=[agent.agent_type for agent in self.agents],
                        current_synthesis=synthesis,
                        current_round=current_round,
                        max_rounds=self.max_rounds
                    )
                    print("No consensus reached, continuing to next round")
                    # If max rounds reached, use the last round's decision as final
                    final_decision = decision
                
        # If no final decision yet, use the last decision
        if not final_decision:
            final_decision = decision

        print(f"Final decision: {final_decision.get('answer', '')}")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Add final decision to history
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time

        return case_history

def parse_structured_output(response_text: str) -> Dict[str, str]:
    try:
        # Try parsing as JSON
        parsed = json.loads(preprocess_response_string(response_text))
        return parsed
    except json.JSONDecodeError:
        # If not valid JSON, extract from text
        # This is a fallback for when the model doesn't format JSON correctly
        lines = response_text.strip().split('\n')
        result = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                result[key] = value

        # Ensure explanation and answer fields exist
        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result

def read_data(base_data_dir: str, base_result_dir: str, repo_filter: list = []) -> List[Dict[str, Any]]:
    datalist = []
    for repo_name in os.listdir(base_data_dir):
        if repo_filter and repo_name not in repo_filter:
            continue
        repo_path = os.path.join(base_data_dir, repo_name)
        rbar = tqdm.tqdm(os.listdir(repo_path), desc=f"Processing {repo_name}", unit="chunk")
        for chunk_dir in rbar:
            result_path = os.path.join(base_result_dir, f'{repo_name}_{chunk_dir}.json')
            if os.path.exists(result_path):
                print(f"Skipping {repo_name}_{chunk_dir} as result already exists")
                continue
            item ={
                'repo_name': repo_name,
                'chunk_dir': chunk_dir,
            }
            datalist.append(item)
    return datalist


def process_item(item, agent_configs, review_model, base_data_dir, base_result_dir):
    merged_contents = []
    try:
        chunk_path = os.path.join(base_data_dir, item['repo_name'], item['chunk_dir'])
        merged_diffs_path = os.path.join(chunk_path, 'tangled_commit')

        merged_file_name = os.listdir(merged_diffs_path)[0]
        merged_contents = json.dumps(load_json(os.path.join(merged_diffs_path, merged_file_name)), ensure_ascii=False)
    except Exception as e:
        print(f"Error loading data for {item['repo_name']}_{item['chunk_dir']}: {e}")
        return
        
    try:
        cucdt = CommitUntangleCoordinator(
            max_rounds=3,
            agent_configs=agent_configs,
            meta_model_key=review_model
        )

        result = cucdt.run(
            qid=f"{item['repo_name']}_{item['chunk_dir']}",
            commit_diff=merged_contents
        )
        item_result = {
            'repo_name': item['repo_name'],
            'chunk_dir': item['chunk_dir'],
            'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'answer': result.get('final_decision', {}).get('answer', None),
            'case_history': result
        }
        save_json(
            item_result,
            os.path.join(base_result_dir, f"{item['repo_name']}_{item['chunk_dir']}.json")
        )
    except Exception as e:
        print(f"Error processing {item['repo_name']}_{item['chunk_dir']}: {e}")

def untangle(base_data_dir, base_result_dir, repo_filter, agent_models, review_model, log_path_dir, max_workers=4):
    datalist = read_data(
        base_data_dir=base_data_dir,
        base_result_dir=base_result_dir,
        repo_filter=repo_filter
    )

    agent_specialties = [AgentType.EXPLICIT, AgentType.IMPLICIT]
    agent_configs = []
    for i, model_name in enumerate(agent_models):
        agent_type = agent_specialties[i % len(agent_specialties)]
        agent_configs.append({
            "agent_id": f"agent_{i+1}",
            "agent_type": agent_type,
            "model_name": model_name
        })

    with tqdm.tqdm(total=len(datalist), desc="Running consultation", unit="case") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in datalist:
                future = executor.submit(process_item, item, agent_configs, review_model, base_data_dir, base_result_dir)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

def evaluate(evaluator: Evaluator):
    evaluator.call_mal()
    return

def main():
    parser = argparse.ArgumentParser(description="Run consultation on datasets")
    parser.add_argument("--add_comments", type=int, required=True, help="Specify whether to add comments (1 for yes, 0 for no)", choices=[0, 1])
    parser.add_argument("--agent_model_config", type=str, required=True, help="Specify model category. Choies: [deepseek-v3]", choices=["deepseek-v3"])
    args = parser.parse_args()

    add_comments = bool(args.add_comments)
    add_graphs = False
    add_file_content = False
    add_concern_number = False
    ma_model_category = args.agent_model_config

    global LLM_MODEL
    LLM_MODEL = LLMType.DEEPSEEK.value


    config = Config(add_comments=add_comments, add_graphs=add_graphs, add_file_content=add_file_content, add_concern_num=add_concern_number, ma_model_category=ma_model_category)
    base_data_dir = config.base_data_dir
    base_result_dir = config.base_ma_result_dir

    os.makedirs(base_result_dir, exist_ok=True)
    repo_filter = config.ma_repo_filter
    agent_model_type = 'deepseek-chat'
    agent_models = [agent_model_type, agent_model_type]
    review_model = agent_model_type

    print(f'Base data directory: {base_data_dir} Base result directory: {base_result_dir}')
    evaluator = Evaluator(base_dir=base_data_dir, result_dir=base_result_dir, evaluation_file_path=re.sub(r'(.*)/[^/]*/[^/]*$', r'\1', base_result_dir) + '/evaluation_result.json')

    untangle(base_data_dir=base_data_dir, base_result_dir=base_result_dir, repo_filter=repo_filter, agent_models=agent_models, review_model=review_model, max_workers=10)
    evaluate(evaluator)

if __name__ == "__main__":
    main()