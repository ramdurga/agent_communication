"""
================================================================================
TUTORIAL 5: Practical Multi-Agent Examples
================================================================================

This tutorial provides complete, runnable examples of multi-agent systems
for real-world use cases:

1. Code Review Team (Developer + Reviewer + Tester)
2. Content Creation Pipeline (Researcher + Writer + Editor)
3. Customer Support System (Classifier + Specialist Agents)
4. Data Analysis Team (Collector + Analyst + Visualizer)

Each example is self-contained and can be run independently.
"""

import operator
from typing import Annotated, TypedDict, Sequence, Literal, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START


def create_llm(temperature: float = 0.7):
    """Create LLM instance."""
    return ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=temperature,
    )


# ============================================================================
# EXAMPLE 1: Code Review Team
# ============================================================================

"""
CODE REVIEW TEAM:
----------------
Simulates a development team reviewing code.

Flow: Developer → Security Reviewer → Code Reviewer → Summary

Each agent specializes in different aspects of code quality.
"""


class CodeReviewState(TypedDict):
    """State for code review system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    code: str
    language: str
    developer_notes: str
    security_review: str
    code_review: str
    test_suggestions: str
    final_summary: str


def create_security_reviewer(llm):
    """Security-focused code reviewer."""

    def reviewer(state: CodeReviewState) -> dict:
        prompt = f"""You are a Security Expert reviewing code for vulnerabilities.

CODE ({state.get('language', 'unknown')}):
```
{state['code']}
```

Review for:
1. SQL injection vulnerabilities
2. XSS vulnerabilities
3. Authentication/authorization issues
4. Data validation problems
5. Sensitive data exposure

Provide specific findings with line references if possible."""

        response = llm.invoke([
            SystemMessage(content="You are a security expert. Find vulnerabilities and suggest fixes."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[SECURITY REVIEWER]:\n{response.content}")],
            "security_review": response.content,
        }

    return reviewer


def create_code_reviewer(llm):
    """Code quality reviewer."""

    def reviewer(state: CodeReviewState) -> dict:
        prompt = f"""You are a Senior Developer reviewing code quality.

CODE ({state.get('language', 'unknown')}):
```
{state['code']}
```

SECURITY FINDINGS (already reviewed):
{state.get('security_review', 'None yet')}

Review for:
1. Code readability and style
2. Performance issues
3. Error handling
4. Best practices
5. Potential bugs

Provide specific, actionable feedback."""

        response = llm.invoke([
            SystemMessage(content="You are a senior developer focused on code quality and best practices."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[CODE REVIEWER]:\n{response.content}")],
            "code_review": response.content,
        }

    return reviewer


def create_test_suggester(llm):
    """Agent that suggests tests for the code."""

    def suggester(state: CodeReviewState) -> dict:
        prompt = f"""You are a QA Engineer suggesting tests for this code.

CODE ({state.get('language', 'unknown')}):
```
{state['code']}
```

Based on the code and reviews:
- Security issues found: {state.get('security_review', 'None')[:200]}
- Code issues found: {state.get('code_review', 'None')[:200]}

Suggest:
1. Unit tests that should be written
2. Edge cases to test
3. Integration test scenarios
4. Security test cases"""

        response = llm.invoke([
            SystemMessage(content="You are a QA engineer who creates comprehensive test plans."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[TEST ENGINEER]:\n{response.content}")],
            "test_suggestions": response.content,
        }

    return suggester


def create_summary_agent(llm):
    """Agent that creates final review summary."""

    def summarizer(state: CodeReviewState) -> dict:
        prompt = f"""Create a final code review summary.

SECURITY REVIEW:
{state.get('security_review', '')}

CODE QUALITY REVIEW:
{state.get('code_review', '')}

TEST SUGGESTIONS:
{state.get('test_suggestions', '')}

Create a concise summary with:
1. Overall assessment (Good/Needs Work/Critical Issues)
2. Top 3 priority fixes
3. Recommended next steps"""

        response = llm.invoke([
            SystemMessage(content="You create clear, actionable code review summaries."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[SUMMARY]:\n{response.content}")],
            "final_summary": response.content,
        }

    return summarizer


def build_code_review_graph(llm):
    """Build the code review pipeline."""
    workflow = StateGraph(CodeReviewState)

    workflow.add_node("security", create_security_reviewer(llm))
    workflow.add_node("quality", create_code_reviewer(llm))
    workflow.add_node("testing", create_test_suggester(llm))
    workflow.add_node("summary", create_summary_agent(llm))

    workflow.add_edge(START, "security")
    workflow.add_edge("security", "quality")
    workflow.add_edge("quality", "testing")
    workflow.add_edge("testing", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()


def example_code_review():
    """Run the code review example."""
    print("\n" + "="*60)
    print("EXAMPLE: Code Review Team")
    print("="*60)

    llm = create_llm(temperature=0.3)  # Lower temp for more consistent reviews
    graph = build_code_review_graph(llm)

    # Sample code to review
    sample_code = '''
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    if result:
        session['user'] = username
        return True
    return False

def get_user_data(user_id):
    data = db.query(f"SELECT * FROM users WHERE id={user_id}")
    return {"password": data.password, "email": data.email, "ssn": data.ssn}
'''

    initial_state = {
        "messages": [],
        "code": sample_code,
        "language": "Python",
        "developer_notes": "",
        "security_review": "",
        "code_review": "",
        "test_suggestions": "",
        "final_summary": "",
    }

    print(f"\nCode to review:\n{sample_code}")
    print("-" * 40)

    result = graph.invoke(initial_state)

    print("\n📝 REVIEW PROCESS:")
    for msg in result["messages"]:
        print(f"\n{msg.content}\n")
        print("-" * 40)

    return result


# ============================================================================
# EXAMPLE 2: Content Creation Pipeline
# ============================================================================

"""
CONTENT CREATION PIPELINE:
-------------------------
Creates high-quality content through multiple specialized agents.

Flow: Topic Analysis → Research → Writing → Editing → Final
"""


class ContentState(TypedDict):
    """State for content creation pipeline."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    target_audience: str
    content_type: str  # blog, article, social, etc.
    outline: str
    draft: str
    edited_content: str
    seo_suggestions: str
    final_content: str


def create_outline_agent(llm):
    """Creates content outline."""

    def outliner(state: ContentState) -> dict:
        prompt = f"""Create a detailed outline for this content:

TOPIC: {state['topic']}
CONTENT TYPE: {state.get('content_type', 'article')}
TARGET AUDIENCE: {state.get('target_audience', 'general')}

Create an outline with:
1. Compelling title options (3)
2. Introduction hook
3. Main sections (3-5)
4. Key points for each section
5. Conclusion approach"""

        response = llm.invoke([
            SystemMessage(content="You are a content strategist who creates engaging outlines."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[OUTLINER]:\n{response.content}")],
            "outline": response.content,
        }

    return outliner


def create_writer_agent(llm):
    """Writes the content draft."""

    def writer(state: ContentState) -> dict:
        prompt = f"""Write content based on this outline:

TOPIC: {state['topic']}
CONTENT TYPE: {state.get('content_type', 'article')}
TARGET AUDIENCE: {state.get('target_audience', 'general')}

OUTLINE:
{state.get('outline', '')}

Write engaging, informative content. Use:
- Clear, concise language
- Examples and analogies
- Smooth transitions
- Strong opening and closing"""

        response = llm.invoke([
            SystemMessage(content="You are a skilled content writer who creates engaging, well-structured content."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[WRITER]:\n{response.content[:500]}...")],
            "draft": response.content,
        }

    return writer


def create_editor_agent(llm):
    """Edits and improves the content."""

    def editor(state: ContentState) -> dict:
        prompt = f"""Edit and improve this draft:

ORIGINAL TOPIC: {state['topic']}
TARGET AUDIENCE: {state.get('target_audience', 'general')}

DRAFT:
{state.get('draft', '')}

Edit for:
1. Grammar and spelling
2. Clarity and flow
3. Engagement and readability
4. Consistency in tone
5. Strong call-to-action

Provide the edited version."""

        response = llm.invoke([
            SystemMessage(content="You are a professional editor who polishes content to perfection."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[EDITOR]: Content edited and polished.")],
            "edited_content": response.content,
            "final_content": response.content,
        }

    return editor


def build_content_pipeline(llm):
    """Build the content creation pipeline."""
    workflow = StateGraph(ContentState)

    workflow.add_node("outline", create_outline_agent(llm))
    workflow.add_node("write", create_writer_agent(llm))
    workflow.add_node("edit", create_editor_agent(llm))

    workflow.add_edge(START, "outline")
    workflow.add_edge("outline", "write")
    workflow.add_edge("write", "edit")
    workflow.add_edge("edit", END)

    return workflow.compile()


def example_content_creation():
    """Run the content creation example."""
    print("\n" + "="*60)
    print("EXAMPLE: Content Creation Pipeline")
    print("="*60)

    llm = create_llm()
    graph = build_content_pipeline(llm)

    initial_state = {
        "messages": [],
        "topic": "How AI is Transforming Small Business Operations",
        "target_audience": "Small business owners",
        "content_type": "blog post",
        "outline": "",
        "draft": "",
        "edited_content": "",
        "seo_suggestions": "",
        "final_content": "",
    }

    print(f"\nTopic: {initial_state['topic']}")
    print(f"Audience: {initial_state['target_audience']}")
    print("-" * 40)

    result = graph.invoke(initial_state)

    print("\n📝 CREATION PROCESS:")
    for msg in result["messages"]:
        print(f"\n{msg.content}\n")
        print("-" * 40)

    print("\n✅ FINAL CONTENT:")
    print(result.get("final_content", "")[:1000])

    return result


# ============================================================================
# EXAMPLE 3: Customer Support System
# ============================================================================

"""
CUSTOMER SUPPORT SYSTEM:
-----------------------
Routes customer queries to specialized support agents.

Flow: Classifier → [Technical | Billing | General] → Response
"""


class SupportState(TypedDict):
    """State for customer support system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_query: str
    category: str  # technical, billing, general
    sentiment: str
    agent_response: str
    follow_up_needed: bool


def create_classifier_agent(llm):
    """Classifies customer queries."""

    def classifier(state: SupportState) -> dict:
        prompt = f"""Classify this customer support query:

QUERY: {state['customer_query']}

Classify into ONE category:
- technical: Software bugs, how-to questions, feature requests
- billing: Payments, refunds, subscription issues
- general: Account questions, feedback, other

Also assess sentiment: positive, neutral, negative, urgent

Respond in format:
CATEGORY: [category]
SENTIMENT: [sentiment]"""

        response = llm.invoke([
            SystemMessage(content="You classify customer queries accurately."),
            HumanMessage(content=prompt)
        ])

        # Parse response
        content = response.content.lower()
        category = "general"
        sentiment = "neutral"

        if "technical" in content:
            category = "technical"
        elif "billing" in content:
            category = "billing"

        if "urgent" in content or "negative" in content:
            sentiment = "urgent"
        elif "positive" in content:
            sentiment = "positive"

        return {
            "messages": [AIMessage(content=f"[CLASSIFIER]: Category={category}, Sentiment={sentiment}")],
            "category": category,
            "sentiment": sentiment,
        }

    return classifier


def create_technical_support(llm):
    """Technical support specialist."""

    def tech_support(state: SupportState) -> dict:
        prompt = f"""You are a Technical Support Specialist.

CUSTOMER QUERY: {state['customer_query']}
SENTIMENT: {state.get('sentiment', 'neutral')}

Provide helpful technical support:
1. Acknowledge the issue
2. Provide clear steps to resolve
3. Offer alternative solutions
4. Be empathetic if frustrated"""

        response = llm.invoke([
            SystemMessage(content="You are a friendly, knowledgeable technical support specialist."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[TECH SUPPORT]:\n{response.content}")],
            "agent_response": response.content,
        }

    return tech_support


def create_billing_support(llm):
    """Billing support specialist."""

    def billing_support(state: SupportState) -> dict:
        prompt = f"""You are a Billing Support Specialist.

CUSTOMER QUERY: {state['customer_query']}
SENTIMENT: {state.get('sentiment', 'neutral')}

Provide helpful billing support:
1. Address their billing concern directly
2. Explain any charges clearly
3. Offer solutions (refund, credit, etc.)
4. Be professional and understanding"""

        response = llm.invoke([
            SystemMessage(content="You are a professional billing support specialist."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[BILLING SUPPORT]:\n{response.content}")],
            "agent_response": response.content,
        }

    return billing_support


def create_general_support(llm):
    """General support agent."""

    def general_support(state: SupportState) -> dict:
        prompt = f"""You are a General Support Agent.

CUSTOMER QUERY: {state['customer_query']}
SENTIMENT: {state.get('sentiment', 'neutral')}

Provide helpful, friendly support:
1. Address their question or concern
2. Provide relevant information
3. Offer to help further
4. Be warm and professional"""

        response = llm.invoke([
            SystemMessage(content="You are a friendly, helpful support agent."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[GENERAL SUPPORT]:\n{response.content}")],
            "agent_response": response.content,
        }

    return general_support


def route_support(state: SupportState) -> Literal["technical", "billing", "general"]:
    """Route to appropriate support agent."""
    category = state.get("category", "general")
    if category == "technical":
        return "technical"
    elif category == "billing":
        return "billing"
    return "general"


def build_support_system(llm):
    """Build customer support system."""
    workflow = StateGraph(SupportState)

    workflow.add_node("classifier", create_classifier_agent(llm))
    workflow.add_node("technical", create_technical_support(llm))
    workflow.add_node("billing", create_billing_support(llm))
    workflow.add_node("general", create_general_support(llm))

    workflow.add_edge(START, "classifier")

    workflow.add_conditional_edges(
        "classifier",
        route_support,
        {
            "technical": "technical",
            "billing": "billing",
            "general": "general",
        }
    )

    workflow.add_edge("technical", END)
    workflow.add_edge("billing", END)
    workflow.add_edge("general", END)

    return workflow.compile()


def example_customer_support():
    """Run the customer support example."""
    print("\n" + "="*60)
    print("EXAMPLE: Customer Support System")
    print("="*60)

    llm = create_llm()
    graph = build_support_system(llm)

    # Test different query types
    queries = [
        "My app keeps crashing when I try to upload photos. It's been happening for 3 days!",
        "I was charged twice for my subscription this month. Please help!",
        "I love your product! Just wanted to say thanks for the great service.",
    ]

    for query in queries:
        print(f"\n{'='*40}")
        print(f"Customer: {query}")
        print("-" * 40)

        state = {
            "messages": [],
            "customer_query": query,
            "category": "",
            "sentiment": "",
            "agent_response": "",
            "follow_up_needed": False,
        }

        result = graph.invoke(state)

        for msg in result["messages"]:
            print(f"\n{msg.content}")

    return result


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TUTORIAL 5: Practical Multi-Agent Examples")
    print("="*60)
    print("\nChoose an example to run:")
    print("1. Code Review Team")
    print("2. Content Creation Pipeline")
    print("3. Customer Support System")

    # Run one example (uncomment others as needed):
    # example_code_review()
    # example_content_creation()
    example_customer_support()
