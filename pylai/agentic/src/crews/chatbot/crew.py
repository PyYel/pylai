import os
from typing import Any
from crewai import Crew, LLM, Process

from ...agents import RAGAgent, OrchestratorAgent, GraphAgent


class ChatbotCrew:
    def __init__(
        self,
        db_host: str = os.getenv("DB_HOST", ""),
        db_password: str = os.getenv("DB_PASSWORD", ""),
        db_user: str = os.getenv("DB_USER", ""),
        INFERENCE_MODEL_NAME: str = os.getenv("INFERENCE_MODEL_NAME", ""),
        INFERENCE_ENDPOINT: str = os.getenv("INFERENCE_ENDPOINT", ""),
        INFERENCE_PROVIDER: str = os.getenv("INFERENCE_PROVIDER", ""),
        INFERENCE_API_KEY: str = os.getenv("INFERENCE_API_KEY", ""),
        step_callback: Any = None,
        task_callback: Any = None,
        **crew_kwargs,
    ):
        self.crew_kwargs = crew_kwargs
        self.step_callback = step_callback
        self.task_callback = task_callback

        chat_llm = LLM(
            model=INFERENCE_MODEL_NAME,
            base_url=INFERENCE_ENDPOINT,
            api_key=INFERENCE_API_KEY,
            temperature=0.7,
            provider=INFERENCE_PROVIDER,
            max_tokens=2048,
            stream=True,
        )
        self.orchestrator_agent = OrchestratorAgent(
            llm=chat_llm,
            verbose=False,
        )

        agent_llm = LLM(
            model=INFERENCE_MODEL_NAME,
            base_url=INFERENCE_ENDPOINT,
            api_key=INFERENCE_API_KEY,
            temperature=0.7,
            provider=INFERENCE_PROVIDER,
            max_tokens=2048,
            stream=True,
        )
        self.rag_agent = RAGAgent(
            db_host=db_host,
            db_password=db_password,
            db_user=db_user,
            llm=agent_llm,
            verbose=False,
        )
        self.graph_agent = GraphAgent(
            db_host=db_host,
            db_password=db_password,
            db_user=db_user,
            llm=agent_llm,
            verbose=False,
        )


    def kickoff(self, user_input: str):

        orchestrator_task = self.orchestrator_agent.create_chat_task(user_input)

        crew = Crew(
            agents=[self.orchestrator_agent, self.rag_agent, self.graph_agent],
            tasks=[orchestrator_task],
            process=Process.sequential,
            stream=True,
            task_callback=self.task_callback,
            **self.crew_kwargs,
        )

        return crew.kickoff(inputs={"user_input": user_input})
