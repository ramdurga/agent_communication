#!/usr/bin/env python3
"""
Analyst Agent - Analyzes research and extracts insights.

Run multiple instances:
    python -m agents.analyst --port 8002 --instance analyst-1
    python -m agents.analyst --port 8012 --instance analyst-2
"""
import argparse
import time
from datetime import datetime
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_agent import BaseAgent
from core.config import LLMConfig
from core.models import TaskRequest, TaskResponse


class AnalystAgent(BaseAgent):
    """
    Analyst Agent - Second in the pipeline.

    Responsibilities:
    - Analyze sentiment of research notes
    - Summarize key points
    - Extract insights and patterns
    """

    def __init__(
        self,
        port: int = 8002,
        instance_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        registry_url: Optional[str] = None,
    ):
        super().__init__(
            agent_type="analyst",
            port=port,
            instance_id=instance_id,
            llm_config=llm_config,
            registry_url=registry_url,
        )

    async def _process(self, request: TaskRequest) -> TaskResponse:
        """Process analysis task."""
        start_time = time.time()

        self.logger.info(f"   📚 Research notes: {len(request.research_notes or '')} chars")

        # Execute tools
        tool_results, tools_used, steps_performed = self._execute_tools(request)

        # Build prompt with tool results
        self.logger.info(f"")
        self.logger.info(f"   ┌─ Generating Analysis with LLM")
        self.logger.info(f"   │  🤖 Model: {self.llm_config.model}")

        prompt = f"""Analyze the following research:

ORIGINAL TASK: {request.task}

RESEARCH NOTES:
{request.research_notes}

TOOL ANALYSIS RESULTS:
- {tool_results.get('analyze_sentiment', 'No sentiment analysis')}
- {tool_results.get('extract_keywords', 'No keywords')}

KEY POINTS IDENTIFIED:
{tool_results.get('summarize_points', 'No summary')}

Provide a structured analysis with:
1. Key insights
2. Patterns identified
3. Recommendations or conclusions"""

        # Call LLM
        llm_start = time.time()
        self.logger.info(f"   │  📥 Prompt length: {len(prompt)} chars")

        llm_output = await self._call_llm_async(prompt)

        llm_time = (time.time() - llm_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # Log thinking if present
        if llm_output.thinking:
            self.logger.info(f"   │  💭 Thinking: {len(llm_output.thinking)} chars")
            steps_performed.append(f"Model reasoning: {len(llm_output.thinking)} chars")

        self.logger.info(f"   │  📤 Answer: {len(llm_output.answer)} chars")
        self.logger.info(f"   │  ⏱️  Time: {llm_time:.0f}ms")
        self.logger.info(f"   └─ ✓ Complete")

        steps_performed.append(f"Generated analysis ({len(llm_output.answer)} chars)")

        # Summary
        self._log_summary(tools_used, llm_time, total_time, llm_output)

        return TaskResponse(
            task_id=request.task_id,
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            instance_id=self.instance_id,
            status="completed",
            result=llm_output.answer,
            thinking=llm_output.thinking,
            next_agent=self.config.next_agent,
            output_field=self.config.output_field,
            processing_time_ms=total_time,
            llm_tokens_used=len(prompt.split()) + len(llm_output.answer.split()),
            hostname=self.hostname,
            timestamp=datetime.now().isoformat(),
            tools_used=tools_used,
            steps_performed=steps_performed
        )

    def _log_summary(self, tools_used, llm_time, total_time, llm_output):
        """Log processing summary."""
        self.logger.info(f"")
        self.logger.info(f"📊 PROCESSING SUMMARY:")
        self.logger.info(f"   • Tools: {len(tools_used)}")
        for tu in tools_used:
            self.logger.info(f"     - {tu.tool_name}: {tu.execution_time_ms:.0f}ms")
        self.logger.info(f"   • LLM: {llm_time:.0f}ms")
        if llm_output.thinking:
            self.logger.info(f"   • Thinking: {len(llm_output.thinking)} chars")
        self.logger.info(f"   • Total: {total_time:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Analyst Agent")
    parser.add_argument("--port", "-p", type=int, default=8002, help="Port")
    parser.add_argument("--instance", "-i", type=str, default=None, help="Instance ID")
    parser.add_argument("--llm-host", type=str, default="localhost", help="Ollama host")
    parser.add_argument("--llm-port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--model", type=str, default="llama3.2", help="LLM model")
    parser.add_argument("--registry", type=str, default=None, help="Registry URL")

    args = parser.parse_args()

    llm_config = LLMConfig(
        host=args.llm_host,
        port=args.llm_port,
        model=args.model
    )

    agent = AnalystAgent(
        port=args.port,
        instance_id=args.instance,
        llm_config=llm_config,
        registry_url=args.registry
    )
    agent.run()


if __name__ == "__main__":
    main()
