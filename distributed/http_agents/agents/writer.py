#!/usr/bin/env python3
"""
Writer Agent - Composes final output from research and analysis.

Run multiple instances:
    python -m agents.writer --port 8003 --instance writer-1
    python -m agents.writer --port 8013 --instance writer-2
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


class WriterAgent(BaseAgent):
    """
    Writer Agent - Final stage in the pipeline.

    Responsibilities:
    - Review research and analysis
    - Format and structure content
    - Compose final comprehensive response
    """

    def __init__(
        self,
        port: int = 8003,
        instance_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        registry_url: Optional[str] = None,
    ):
        super().__init__(
            agent_type="writer",
            port=port,
            instance_id=instance_id,
            llm_config=llm_config,
            registry_url=registry_url,
        )

    async def _process(self, request: TaskRequest) -> TaskResponse:
        """Process writing task."""
        start_time = time.time()

        self.logger.info(f"   📚 Research notes: {len(request.research_notes or '')} chars")
        self.logger.info(f"   🔍 Analysis: {len(request.analysis or '')} chars")

        # Execute tools
        tool_results, tools_used, steps_performed = self._execute_tools(request)

        # Build prompt with all inputs
        self.logger.info(f"")
        self.logger.info(f"   ┌─ Composing Final Response with LLM")
        self.logger.info(f"   │  🤖 Model: {self.llm_config.model}")

        prompt = f"""Create a final response based on:

ORIGINAL TASK: {request.task}

RESEARCH NOTES:
{request.research_notes}

ANALYSIS:
{request.analysis}

SUMMARY OF KEY POINTS:
{tool_results.get('summarize_points', 'No summary')}

Write a comprehensive, well-structured final response that:
1. Directly addresses the original task
2. Incorporates key findings from research
3. Includes insights from the analysis
4. Is clear, concise, and well-organized"""

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

        steps_performed.append(f"Generated final response ({len(llm_output.answer)} chars)")

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
            next_agent=self.config.next_agent,  # None for writer
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
    parser = argparse.ArgumentParser(description="Writer Agent")
    parser.add_argument("--port", "-p", type=int, default=8003, help="Port")
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

    agent = WriterAgent(
        port=args.port,
        instance_id=args.instance,
        llm_config=llm_config,
        registry_url=args.registry
    )
    agent.run()


if __name__ == "__main__":
    main()
