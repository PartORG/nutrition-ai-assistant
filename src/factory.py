"""
factory - Composition root for the nutrition AI assistant.

ALL dependency wiring happens here. No other module constructs its own
dependencies. Adapters (CLI, REST, Flutter) call this factory to get
fully configured services and agents.

Usage:
    from factory import ServiceFactory
    from infrastructure.config import Settings

    config = Settings.from_env()
    factory = ServiceFactory(config)
    await factory.initialize()  # one-time startup

    # For direct service access (REST API):
    service = factory.create_recommendation_service()
    result = await service.get_recommendations(ctx, query)

    # For conversational agent (CLI, WebSocket):
    agent = factory.create_agent(ctx)
    response = await agent.run(ctx, user_input)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from infrastructure.config import Settings
from infrastructure.persistence.connection import AsyncSQLiteConnection
from infrastructure.persistence.migrations import run_migrations
from infrastructure.persistence.user_repo import SQLiteUserRepository
from infrastructure.persistence.medical_repo import SQLiteMedicalRepository
from infrastructure.persistence.recipe_repo import SQLiteRecipeRepository
from infrastructure.persistence.nutrition_repo import SQLiteNutritionRepository
from infrastructure.persistence.profile_repo import SQLiteProfileRepository
from infrastructure.persistence.conversation_repo import SQLiteConversationRepository
from infrastructure.persistence.chat_message_repo import SQLiteChatMessageRepository
from infrastructure.persistence.auth_repo import SQLiteAuthenticationRepository
from infrastructure.persistence.analytics_repo import SQLiteAnalyticsRepository
from infrastructure.llm.intent_parser import OllamaIntentParser
from infrastructure.llm.safety_filter import OllamaSafetyFilter
from infrastructure.rag.medical_rag import MedicalRAG
from infrastructure.rag.recipe_rag import RecipeNutritionRAG
from infrastructure.cnn.ingredient_detector import LLaVAIngredientDetector
from infrastructure.cnn.yolo_service_detector import YOLOServiceDetector
from infrastructure.cnn.fallback_detector import FallbackIngredientDetector
from application.context import SessionContext
from application.services.recommendation import RecommendationService
from application.services.recipe_manager import RecipeManagerService
from application.services.profile import ProfileService
from application.services.image_analysis import ImageAnalysisService
from application.services.chat_history import ChatHistoryService
from application.services.authentication import AuthenticationService
from agent.tools.registry import ToolRegistry
from agent.tools.search_recipes import SearchRecipesTool
from agent.tools.save_recipe import SaveRecipeTool
from agent.tools.analyze_image import AnalyzeImageTool
from agent.tools.show_recipe import ShowRecipeTool
from agent.tools.general_chat import GeneralChatTool
from agent.tools.nutrition_status import NutritionStatusTool
from agent.tools.safety_guard import SafetyGuardTool
from agent.tools.crisis_support import CrisisSupportTool
from agent.memory import ConversationMemory
from agent.prompt import build_system_prompt
from agent.executor import AgentExecutor

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Composition root — wires all dependencies together.

    Call initialize() once at startup, then create services/agents as needed.
    Expensive LLM objects are built once during initialize() and reused.
    """

    def __init__(self, config: Settings):
        self._config = config
        self._connection = AsyncSQLiteConnection(config.db_path)

        # Lazy singletons for expensive resources
        self._medical_rag: Optional[MedicalRAG] = None
        self._recipe_rag: Optional[RecipeNutritionRAG] = None
        self._agent_llm = None
        self._intent_parser: Optional[OllamaIntentParser] = None
        self._safety_filter: Optional[OllamaSafetyFilter] = None
        self._image_detector = None

        self._initialized = False

    async def initialize(self) -> None:
        """One-time startup: run migrations, initialize RAG systems.

        Must be called before creating services or agents.
        """
        logger.info("Initializing ServiceFactory...")

        await run_migrations(self._connection)
        logger.info("Database migrations complete")

        # ── Build shared LLM-backed components ONCE ──────────────────
        self._intent_parser = OllamaIntentParser(
            model_name=self._config.llm_model,
            ollama_base_url=self._config.ollama_base_url,
        )
        self._safety_filter = OllamaSafetyFilter(
            model_name=self._config.llm_model,
            ollama_base_url=self._config.ollama_base_url,
        )
        self._agent_llm = self._build_agent_llm()
        self._image_detector = self._build_image_detector()
        logger.info("Shared LLM components initialized")

        # Initialize Medical RAG with configured provider
        self._medical_rag = MedicalRAG(
            folder_paths=[str(self._config.pdf_dir)],
            model_name=self._config.rag_llm_model,
            vectorstore_path=str(self._config.medical_vectorstore_path),
            ollama_base_url=self._config.ollama_base_url,
            llm_provider=self._config.rag_llm_provider,
            openai_api_key=self._config.openai_api_key,
            groq_api_key=self._config.groq_api_key,
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._medical_rag.initialize, False,
        )
        logger.info("Medical RAG initialized (provider=%s)", self._config.rag_llm_provider)

        # Initialize Recipe RAG with configured provider
        self._recipe_rag = RecipeNutritionRAG(
            data_folder=str(self._config.data_dir),
            model_name=self._config.rag_llm_model,
            vectorstore_path=str(self._config.recipes_nutrition_vector_path),
            ollama_base_url=self._config.ollama_base_url,
            llm_provider=self._config.rag_llm_provider,
            openai_api_key=self._config.openai_api_key,
            groq_api_key=self._config.groq_api_key,
        )
        await loop.run_in_executor(
            None, self._recipe_rag.initialize,
        )
        logger.info("Recipe RAG initialized (provider=%s)", self._config.rag_llm_provider)

        self._initialized = True
        logger.info("ServiceFactory ready")

    # ------------------------------------------------------------------
    # Service creation
    # ------------------------------------------------------------------

    def create_recommendation_service(self) -> RecommendationService:
        """Create a RecommendationService with all dependencies wired.

        Reuses the cached intent_parser and safety_filter singletons.
        """
        self._ensure_initialized()
        return RecommendationService(
            intent_parser=self._intent_parser,
            medical_rag=self._medical_rag,
            recipe_rag=self._recipe_rag,
            safety_filter=self._safety_filter,
            medical_repo=SQLiteMedicalRepository(self._connection),
            nutrition_repo=SQLiteNutritionRepository(self._connection),
        )

    def create_recipe_manager(self) -> RecipeManagerService:
        """Create a RecipeManagerService."""
        return RecipeManagerService(
            recipe_repo=SQLiteRecipeRepository(self._connection),
            nutrition_repo=SQLiteNutritionRepository(self._connection),
        )

    def create_profile_service(self) -> ProfileService:
        """Create a ProfileService."""
        return ProfileService(
            profile_repo=SQLiteProfileRepository(self._connection),
            medical_repo=SQLiteMedicalRepository(self._connection),
        )

    def create_image_analysis_service(self) -> ImageAnalysisService:
        """Create an ImageAnalysisService.

        Reuses the cached image detector singleton.
        """
        rec_service = self.create_recommendation_service()
        return ImageAnalysisService(
            detector=self._image_detector,
            recommendation_service=rec_service,
        )

    def create_authentication_service(self) -> AuthenticationService:
        """Create an AuthenticationService with all dependencies wired."""
        return AuthenticationService(
            user_repo=SQLiteUserRepository(self._connection),
            auth_repo=SQLiteAuthenticationRepository(self._connection),
            jwt_secret=self._config.jwt_secret,
            jwt_expiry_hours=self._config.jwt_expiry_hours,
        )

    def create_chat_history_service(self) -> ChatHistoryService:
        """Create a ChatHistoryService for conversation persistence."""
        return ChatHistoryService(
            conversation_repo=SQLiteConversationRepository(self._connection),
            message_repo=SQLiteChatMessageRepository(self._connection),
        )

    def create_user_repository(self) -> SQLiteUserRepository:
        """Return a user repository for direct user entity lookups."""
        return SQLiteUserRepository(self._connection)

    def create_analytics_repository(self) -> SQLiteAnalyticsRepository:
        """Return an analytics repository for aggregate read queries."""
        return SQLiteAnalyticsRepository(self._connection)

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def create_agent(self, ctx: SessionContext) -> AgentExecutor:
        """Create a fully configured AgentExecutor for a session.

        Only lightweight objects (tools, memory, registry) are created
        per call — the LLM itself is reused from the cache.

        Args:
            ctx: Session context with user info.

        Returns:
            AgentExecutor ready for chat with DB-backed conversation history.
        """
        self._ensure_initialized()

        rec_service = self.create_recommendation_service()
        recipe_manager = self.create_recipe_manager()
        image_service = self.create_image_analysis_service()
        chat_history_service = self.create_chat_history_service()

        # Register tools
        nutrition_repo = SQLiteNutritionRepository(self._connection)
        medical_repo = SQLiteMedicalRepository(self._connection)

        registry = ToolRegistry()
        registry.register(SearchRecipesTool(rec_service))
        registry.register(SaveRecipeTool(recipe_manager))
        registry.register(ShowRecipeTool())
        registry.register(AnalyzeImageTool(image_service))
        registry.register(GeneralChatTool())
        registry.register(NutritionStatusTool(nutrition_repo, medical_repo))
        registry.register(SafetyGuardTool())
        registry.register(CrisisSupportTool())

        # Build system prompt with registered tools
        system_prompt = build_system_prompt(registry)

        return AgentExecutor(
            llm=self._agent_llm,
            tools=registry,
            memory=ConversationMemory(
                chat_history_service=chat_history_service,
            ),
            system_prompt=system_prompt,
            max_iterations=self._config.agent_max_iterations,
            chat_history_service=chat_history_service,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_agent_llm(self):
        """Build the LLM for the conversational agent (called once during init)."""
        provider = self._config.agent_llm_provider
        if provider == "groq":
            from langchain_groq import ChatGroq
            logger.info("Agent LLM: Groq (%s)", self._config.agent_llm_model)
            return ChatGroq(
                model=self._config.agent_llm_model,
                temperature=0,
                groq_api_key=self._config.groq_api_key,
                max_tokens=512,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            logger.info("Agent LLM: OpenAI (%s)", self._config.agent_llm_model_openai)
            return ChatOpenAI(
                model=self._config.agent_llm_model_openai,
                temperature=0,
                openai_api_key=self._config.openai_api_key,
            )
        else:
            from langchain_ollama import ChatOllama
            logger.info("Agent LLM: Ollama (%s)", self._config.agent_llm_model)
            return ChatOllama(
                model=self._config.agent_llm_model,
                temperature=0,
                base_url=self._config.ollama_base_url,
            )

    def _build_image_detector(self):
        """Build the ingredient detector (called once during init)."""
        detector_type = self._config.cnn_detector_type
        llava_model = self._config.cnn_model_path or "llava"

        if detector_type == "llava_only":
            detector = LLaVAIngredientDetector(
                ollama_base_url=self._config.ollama_base_url,
                model=llava_model,
            )
            logger.info(
                "Image detector: LLaVA ('%s') at %s",
                llava_model, self._config.ollama_base_url,
            )
        elif detector_type == "yolo_only":
            detector = YOLOServiceDetector(
                service_url=self._config.yolo_service_url,
            )
            logger.info(
                "Image detector: YOLO-only at %s", self._config.yolo_service_url,
            )
        else:  # "yolo_with_fallback" (default)
            yolo = YOLOServiceDetector(
                service_url=self._config.yolo_service_url,
            )
            llava = LLaVAIngredientDetector(
                ollama_base_url=self._config.ollama_base_url,
                model=llava_model,
            )
            detector = FallbackIngredientDetector(primary=yolo, fallback=llava)
            logger.info(
                "Image detector: YOLO (%s) with LLaVA fallback",
                self._config.yolo_service_url,
            )
        return detector

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "ServiceFactory not initialized. Call await factory.initialize() first."
            )
