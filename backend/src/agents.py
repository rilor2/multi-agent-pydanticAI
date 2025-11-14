"""
Multi-Agent System for Web Interface

This module defines a multi-agent system with specialized agents for:
- Code generation and execution
- Web search
- Image analysis

Adapted from the multi_agent_demo.py file for web use with FastAPI.
"""

import logging
import os
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import httpx
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, ImageUrl, BinaryContent
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits

import dotenv

dotenv.load_dotenv()

model = GeminiModel(
    "gemini-2.5-flash",
    provider=GoogleVertexProvider(),
)


# ===================================================
# DEPENDENCY INJECTION WITH RUNCONTEXT
# ===================================================
@dataclass
class SharedContext:
    """Shared context across all agents"""

    username: str
    preferences: dict[str, str]
    session_id: str
    http_client: httpx.AsyncClient
    current_image_path: Optional[str] = None  # Add current image path


# ===================================================
# SPECIALIZED AGENT OUTPUT STRUCTURES
# ===================================================
class CodeResult(BaseModel):
    """Result from the coding agent"""

    code: str = Field(description="Generated or fixed code")
    explanation: str = Field(description="Explanation of the code")
    execution_result: str = Field(
        description="Only include the content found within the <output>...</output> tags from the code execution result. Do not include <status>, <dependencies>, or any other tags."
    )


class SearchResult(BaseModel):
    """Result from the web search agent"""

    answer: str = Field(description="Comprehensive summary of the search results.")
    sources: list[str] = Field(
        description="Sources used for the answer", default_factory=list
    )


class ImageAnalysisResult(BaseModel):
    """Result from the image analysis"""

    description: str = Field(description="Detailed description of the image")
    objects: list[str] = Field(
        description="Objects detected in the image", default_factory=list
    )
    scene_type: str = Field(description="Type of scene (indoor, outdoor, etc.)")

class OrderResult(BaseModel):
    """Result from the order placement"""
    description: str = Field(description="Detailed instruction for checkout")
    client_details: str = Field(description="name, surname and email")
    delivery_details: str = Field(description="delivery address")
    payment_method: str = Field(description="payment method to use", default="EXPRESS") 
    payment_requested: bool = Field(description="payment requested", default=False) 

class CatalogResult(BaseModel):
    """Result from the catalog search"""
    description: str = Field(description="Detailed description of the object to search")
    objects: list[str] = Field(
        description="Furniture names according to Ikea", default_factory=list
    )

class MainAgentResult(BaseModel):
    """Overall result from the main agent"""

    answer: str = Field(
        description="The complete answer to the user's question, combining the results of each agents response."
    )
    search_result: SearchResult = Field(
        description="Result from the web search agent if used", default=None
    )
    image_analysis_result: ImageAnalysisResult = Field(
        description="Result from the image analysis agent if used", default=None
    )
    code_result: CodeResult = Field(
        description="Result from the Python code agent if used", default=None
    )
    catalog_result: CatalogResult = Field(
        description="Result from the catalog agent if used", default=None
    )
    order_result: OrderResult = Field(
        description="Result from the order agent if used", default=None
    ) 



# ===================================================
# MCP SERVER SETUP (SHARED ACROSS AGENTS)
# ===================================================
mcp_pydantic = MCPServerStdio(
    command="deno",
    args=[
        "run",
        "-N",
        "-R=node_modules",
        "-W=node_modules",
        "--node-modules-dir=auto",
        "jsr:@pydantic/mcp-run-python",
        "stdio",
    ],
)

mcp_fetch = MCPServerStdio(
    command="uvx",
    args=["mcp-server-fetch"],
)


# ===================================================
# SPECIALIZED AGENTS
# ===================================================
# 1. Python Code Agent
code_agent = Agent(
    model=model,
    mcp_servers=[mcp_pydantic],
    deps_type=SharedContext,
    output_type=CodeResult,
    retries=3,
    system_prompt=(
        "You are a coding expert that specializes in Python programming.\n\n"
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "- You MUST provide your answer in the structured CodeResult format\n"
        "- The 'code' field must contain clean, working code\n"
        "- The 'explanation' field should explain how the code works\n"
        "- If you run the code, include the result in 'execution_result'\n"
    ),
)

# 3. Image Analysis Agent
image_agent = Agent(
    model=model,
    deps_type=SharedContext,
    output_type=ImageAnalysisResult,
    retries=3,
    system_prompt=(
        "You are an image analysis expert. You will receive binary image data to analyze.\n\n"
        "When you receive the binary image data:\n"
        "1. Analyze the visible elements in the image\n"
        "2. Provide a detailed description\n"
        "3. List the objects you can identify\n"
        "4. Determine the scene type\n"
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "- You MUST provide your answer in the structured ImageAnalysisResult format\n"
        "- The 'description' field must contain a detailed description of the image\n"
        "- The 'objects' field should list the key objects detected in the image\n"
        "- The 'scene_type' field should describe the type of scene (indoor, outdoor, etc.)\n"
    ),
)

catalog_agent = Agent(
    model=model, 
    deps_type=SharedContext,
    output_type=CatalogResult,
    retries=3,
    system_prompt=(
        "You are an Ikea sales expert. You will receive the name of a piece of furniture to look for.\n\n"
        "When you receive name of the furniture:\n"
        "1. Analyze the name\n"
        "2. Provide a detailed description from the Ikea website\n"
        "3. List the option you can get\n"
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "- You MUST provide your answer in the structured CatalogResult format\n"
        "- The 'description' field must contain a detailed description of the furniture\n"
        "- The 'objects' field should list possible furniture candidates\n"
    ), 
)

order_agent = Agent(
    model=model,
    deps_type=SharedContext,
    output_type=OrderResult,
    retries=3,
    system_prompt=(
        "You are an Order system. You will receive the ikea name of the furniture to buy.\n\n"
        "When you receive name of the furniture, we simulate opening an order in our system, payment method will be EXPRESS:\n"
        "1. Provide user details, name, surname and email\n"
        "2. Provide delivery destination"
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "- You MUST provide your answer in the structured OrderResult format\n"
        "- The 'client_details' field should contain user details, name, surname and email, space separated\n"
        "- The 'delivery_details' field should contain user details, name, surname and email, space separated\n"
        "- The 'payment_method' field will always be EXPRESS, NOT ANY OTHER value\n" 

    ),  
)
# ===================================================
# IMAGE HANDLING FUNCTIONS (DIRECT UTILITIES)
# ===================================================
async def analyze_image_file(file_path: str, ctx: RunContext) -> ImageAnalysisResult:
    """Analyze an image file directly without delegation to the image agent.

    This function handles image file analysis directly, avoiding the multiple attempts
    issue that was occurring in the original code.
    """
    if not os.path.exists(file_path):
        return ImageAnalysisResult(
            description=f"Error: File {file_path} does not exist.",
            objects=[],
            scene_type="unknown",
        )

    try:
        # Create binary content from file
        content_type = (
            "image/jpeg" if file_path.endswith((".jpg", ".jpeg")) else "image/png"
        )
        messages = [
            "Analyze this image in detail:",
            BinaryContent(data=Path(file_path).read_bytes(), media_type=content_type),
        ]

        # Self-call with the prepared image
        result = await image_agent.run(
            messages,
            deps=ctx.deps,
            usage=ctx.usage,
        )

        # Clean up the file after analysis
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {file_path}: {e}")

        return result.output
    except Exception as e:
        # Try to clean up even if analysis failed
        try:
            os.remove(file_path)
        except OSError:
            logging.error(f"Error analyzing image from file {file_path}: {str(e)}")

        return ImageAnalysisResult(
            description=f"Error analyzing image from file {file_path}: {str(e)}",
            objects=[],
            scene_type="unknown",
        )


# ===================================================
# MAIN AGENT
# ===================================================
main_agent = Agent(
    model=model,
    mcp_servers=[mcp_pydantic, mcp_fetch],  # Share the same MCP server
    deps_type=SharedContext,
    output_type=MainAgentResult,
    retries=3,
    end_strategy="early",
    system_prompt=(
        "You are an AI assistant that can handle a variety of tasks by delegating to specialized agents.\n\n"
        "WHEN TO USE SPECIALIZED TOOLS:\n"
        "1. Python Code Expert (code_expert tool):\n"
        "   - When user explicitly asks for Python code\n"
        "   - When user needs help with Python programming\n"
        "   - When user asks about Python code execution or debugging\n"
        "   - NOTE: This expert ONLY supports Python code\n\n" 
        "2. Catalog expert(catalog_expert tool):\n"
        "   - When user explicitly asks for buying a furniture\n"
        "   - When user needs help with furniture ordering \n"
        "   - When users has ANY questions about house, furnitures, office or anything else"
        "   - NOTE: we are Ikea biased, we ONLY sell ikea products"
        "3. Order expert(order_expert tool):\n"
        "   - When user explicitly asks for ordering a specific Ikea furniture by Ikea name\n"
        "   - When user select an Ikea item to buy or to get\n"
        "   - When user confirms they want to proceed with the order\n"
        "IMPORTANT: For general conversation, greetings, or simple questions, DO NOT use any specialized tools.\n\n"
        "CRITICAL OUTPUT INSTRUCTION:\n"
        "The user will ONLY see the 'answer' field of your response.\n"
        "You MUST always compose a fully self-contained answer in the 'answer' field, summarizing and including all important details from any sub-agent results (code, search, image).\n"
        "NEVER assume the user can see the structured output or any fields other than 'answer'.\n"
        "If you use a specialized agent, always clearly explain its results in the 'answer' field, including code, explanations, search findings, and image analysis as appropriate.\n\n"
        "When you use a specialized agent, you MUST also include its result in the corresponding field (search_result, code_result, catalog_result, order_result or image_analysis_result) of the MainAgentResult object, as well as summarizing it in the answer field. Never leave these fields empty if you used a sub-agent.\n\n"
        "Once you display a list of furniture, ask the user if he wants to proceed with the order"
    ),
)


# ===================================================
# DYNAMIC SYSTEM PROMPT WITH RUNCONTEXT
# ===================================================
@main_agent.system_prompt
def personalized_context(ctx: RunContext[SharedContext]) -> str:
    """Dynamic system prompt that uses RunContext to access dependencies"""
    today = date.today().strftime("%Y-%m-%d")
    base_prompt = (
        f"Today's date is {today}.\n"
        f"You are speaking with {ctx.deps.username}, who prefers {ctx.deps.preferences['style']} responses.\n"
        f"This is session {ctx.deps.session_id}."
    )    
    return base_prompt


# ===================================================
# DELEGATION TOOLS ON MAIN AGENT
# ===================================================
@main_agent.tool
async def code_expert(ctx: RunContext[SharedContext], coding_task: str) -> CodeResult:
    """Delegate Python coding tasks to the specialized Python code agent.

    Args:
        coding_task: The Python coding task description

    Note:
        This expert only supports Python code.
    """
    result = await code_agent.run(
        coding_task,
        deps=ctx.deps,  # Pass the same deps
        usage=ctx.usage,  # Share usage tracking
    )
    return result.output



@main_agent.tool
async def image_expert(
    ctx: RunContext[SharedContext], user_query: str
) -> ImageAnalysisResult:
    """Analyze an image specified in the context or user query.

    Args:
        user_query: The complete user request about image analysis
    """
    # If we have a current image path, use that directly
    if ctx.deps.current_image_path:
        image_path = ctx.deps.current_image_path

        if not os.path.dirname(image_path):
            # Create the full path to the uploads directory
            uploads_dir = Path(__file__).parent.parent.parent / "uploads"
            image_path = str(uploads_dir / image_path)

        return await analyze_image_file(image_path, ctx)

    # Handle URL case for backwards compatibility, but current_image_path is the primary mechanism
    if "http://" in user_query or "https://" in user_query:
        # Extract URL from query
        words = user_query.split()
        for word in words:
            if word.startswith(
                ("http://", "https://")
            ):  # Forward to image agent with ImageUrl
                messages = ["Analyze this image in detail:", ImageUrl(url=word)]
                try:
                    result = await image_agent.run(
                        messages,
                        deps=ctx.deps,
                        usage=ctx.usage,
                    )
                    return result.output
                except Exception as e:
                    return ImageAnalysisResult(
                        description=f"Error analyzing image from URL: {str(e)}",
                        objects=[],
                        scene_type="unknown",
                    )

    # No image path or URL found
    return ImageAnalysisResult(
        description="No image was found to analyze. Please upload an image or provide a valid image URL.",
        objects=[],
        scene_type="unknown",
    )

@main_agent.tool
async def catalog_expert(
        ctx: RunContext[SharedContext], 
        user_query: str
) -> CatalogResult:
    try: 
        result = await catalog_agent.run(
                        user_query,
                        deps=ctx.deps,
                        usage=ctx.usage,
                )
        print(result.output)
        return result.output
    except Exception as e:
       return CatalogResult(
            description=f"Error analysing catalog items: {str(e)}",
            objects= []

        ) 

@main_agent.tool
async def order_expert(
    ctx: RunContext[SharedContext], user_query: str
) -> OrderResult:
    try:
        print(user_query)
        
        message = (
            f"Generate an order template with random client_details, "
            f"delivery_details, and payment method set to EXPRESS."
            f"ONLY WHERN the user confirms the order, set payment_requested to True, and ask the User to pay, otherwise payment_requested is False."
        )
        result = await order_agent.run(
            message,
            deps=ctx.deps,
            usage=ctx.usage,
        )
        print(result.output)
        return result.output

    except Exception as e:
       return CatalogResult(
            description=f"Error analysing catalog items: {str(e)}",
            objects= []

        ) 


# ===================================================
# STREAMING AGENT
# ===================================================
main_agent_stream = Agent(
    model=model,
    mcp_servers=[mcp_pydantic, mcp_fetch],  # Share the same MCP server
    deps_type=SharedContext,
    output_type=str,  # Use str for streaming
    retries=3,
    system_prompt=(
        "You are an AI assistant that can handle a variety of tasks by delegating to specialized agents.\n\n"
        "WHEN TO USE SPECIALIZED TOOLS:\n"
        "1. Python Code Expert (code_expert tool):\n"
        "   - When user explicitly asks for Python code\n"
        "   - When user needs help with Python programming\n"
        "   - When user asks about Python code execution or debugging\n"
        "   - NOTE: This expert ONLY supports Python code\n\n"
        "2. Catalog expert(catalog_expert tool):\n"
        "   - When user explicitly asks for buying a furniture\n"
        "   - When user needs help with furniture ordering \n"
        "   - When users has ANY questions about house, furnitures, office or anything else"
        "   - NOTE: we are Ikea biased, we ONLY sell ikea products"
        "3. Order expert(order_expert tool):\n"
        "   - When user explicitly asks for ordering a specific Ikea furniture by Ikea name\n"
        "   - When user select an Ikea item to buy or to get\n"
        "   - When user confirms they want to proceed with the order\n"
        "IMPORTANT: For general conversation, greetings, or simple questions, DO NOT use any specialized tools.\n"
        "Just respond naturally using your own knowledge.\n\n"
        "FORMATTING INSTRUCTIONS:\n"
        "- Provide your responses in natural, conversational text\n"
        "- When using specialist tools, incorporate their results naturally into your response\n"
        "- Make your responses engaging and informative\n"
        "- Once you display a list of furniture, ask the user if he wants to proceed with the order"
    ),
)


# Add the same tools to the streaming agent
main_agent_stream.system_prompt(personalized_context)
main_agent_stream.tool(code_expert)
main_agent_stream.tool(catalog_expert)
main_agent_stream.tool(order_expert)


# ===================================================
# Functions for Web API Use
# ===================================================
async def process_query(
    prompt: str,
    session_id: str,
    username: str = "User",
    preferences: dict = None,
    message_history: list[ModelMessage] = None,
    usage_limits: UsageLimits = None,
    current_image_path: Optional[str] = None,  # Add image path parameter
) -> Any:
    """Process a query using the multi-agent system.

    This function is designed to be called from the FastAPI routes.

    Args:
        prompt: The user's query
        session_id: Unique identifier for the session
        username: User's name
        preferences: User preferences (style, etc.)
        message_history: Previous conversation history
        usage_limits: Usage limits for the agent run
        current_image_path: Optional path to an uploaded image

    Returns:
        MainAgentResult object containing the response
    """

    async with httpx.AsyncClient() as client:
        shared_context = SharedContext(
            username=username,
            preferences=preferences,
            session_id=session_id,
            http_client=client,
            current_image_path=current_image_path,  # Add image path to context
        )

        usage = Usage()

        async with main_agent.run_mcp_servers():
            result = await main_agent.run(
                prompt,
                deps=shared_context,
                usage=usage,
                usage_limits=usage_limits,
                message_history=message_history,
            )

            return {
                "result": result.output,
                "messages": result.all_messages(),
                "new_messages": result.new_messages(),
                "usage": {
                    "requests": usage.requests,
                    "total_tokens": usage.total_tokens,
                },
            }


# Update the streaming function to use the correct agent
async def process_query_stream(
    prompt: str,
    session_id: str,
    username: str = "User",
    preferences: dict = None,
    message_history: list[ModelMessage] = None,
    usage_limits: UsageLimits = None,
    current_image_path: Optional[str] = None,  # Add image path parameter
):
    """Process a query with streaming response."""
    async with httpx.AsyncClient() as client:
        shared_context = SharedContext(
            username=username,
            preferences=preferences or {"style": "friendly"},
            session_id=session_id,
            http_client=client,
            current_image_path=current_image_path,  # Add image path to context
        )

        usage = Usage()

        async with main_agent_stream.run_mcp_servers():
            async with main_agent_stream.run_stream(
                prompt,
                deps=shared_context,
                usage=usage,
                usage_limits=usage_limits,
                message_history=message_history,
            ) as result:
                async for chunk in result.stream_text():
                    yield {"text": chunk, "done": False}

                # Get final data
                final_data = await result.get_output()

                # For the final message, we don't need to serialize the messages
                # The library will handle that internally
                yield {
                    "text": "",
                    "done": True,
                    "result": final_data,
                    "new_messages": result.new_messages(),  # Let Pydantic handle serialization
                    "usage": {
                        "requests": usage.requests,
                        "total_tokens": usage.total_tokens,
                    },
                }
