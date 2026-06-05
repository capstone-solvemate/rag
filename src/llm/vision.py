from langchain_openai import ChatOpenAI
from src.config import config

from src.utils.logger import get_logger
from langchain_core.messages import HumanMessage, SystemMessage
from src.llm.prompt_templates import VISION_SYSTEM_PROMPT
from src.core.exceptions import VisionAnalysisError
import json

logger = get_logger(__name__)

def _get_vision_model() -> ChatOpenAI:
    """Instantiate the ChatOpenAI client.

    Kept as a private factory function so it can be patched
    cleanly in tests without touching module-level state.

    Returns:
        Configured ChatOpenAI instance using gpt-4o-mini.
    """
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )

async def analyze_images(
    image_urls: list[str],
) -> dict:
    """
    Analyze one or more assembly photos and return structured visual context.

    Args:
        image_urls: List of image URLs or data URLs.

    Returns:
        Parsed JSON dictionary.

    Raises:
        VisionAnalysisError
    """

    logger.info(
        "Analyzing images | count=%s",
        len(image_urls),
    )

    content = [
        {
            "type": "text",
            "text": (
                "Analyze all images and return JSON only. "
                "Do not wrap the JSON in markdown."
            ),
        }
    ]

    for image_url in image_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high",
                },
            }
        )

    messages = [
        SystemMessage(content=VISION_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    try:
        model = _get_vision_model()

        response = await model.ainvoke(messages)

        raw_output = response.content.strip()

        result = json.loads(raw_output)

        logger.info(
            "Vision analysis completed | images=%s",
            len(image_urls),
        )

        return result

    except json.JSONDecodeError as exc:
        logger.error(
            "Vision returned invalid JSON: %s",
            exc,
        )

        raise VisionAnalysisError(
            message="Vision model returned invalid JSON.",
            cause=exc,
        ) from exc

    except Exception as exc:
        logger.error(
            "Vision analysis failed: %s: %s",
            type(exc).__name__,
            exc,
        )

        raise VisionAnalysisError(
            message="Vision analysis failed.",
            cause=exc,
        ) from exc