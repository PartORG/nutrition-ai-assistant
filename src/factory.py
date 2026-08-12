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
import logging

from agent.executor import AgentExecutor
from agent.memory import ConversationMemory
from agent.prompt import build_system_prompt
from agent.tools.analyze_image import AnalyzeImageTool
from agent.tools.crisis_support import CrisisSupportTool
from agent.tools.general_chat import GeneralChatTool
from agent.tools.lookup_nutrition import LookupNutritionTool
from agent.tools.nutrition_status import NutritionStatusTool
from agent.tools.registry import ToolRegistry
from agent.tools.safety_guard import SafetyGuardTool
from agent.tools.save_recipe import SaveRecipeTool
from agent.tools.search_recipes import SearchRecipesTool
from agent.tools.show_recipe import ShowRecipeTool
from application.context import SessionContext
from application.services.authentication import AuthenticationService
from application.services.chat_history import ChatHistoryService
from application.services.image_analysis import ImageAnalysisService
from application.services.profile import ProfileService
from application.services.recipe_manager import RecipeManagerService
from application.services.recommendation import RecommendationService
from infrastructure.cnn.ingredient_detector import LLaVAIngredientDetector
from infrastructure.config import Settings
from infrastructure.llm.intent_parser import IntentParser
from infrastructure.llm.safety_filter import SafetyFilter
from infrastructure.persistence.analytics_repo import SQLiteAnalyticsRepository
from infrastructure.persistence.auth_repo import SQLiteAuthenticationRepository
from infrastructure.persistence.chat_message_repo import SQLiteChatMessageRepository
from infrastructure.persistence.connection import AsyncSQLiteConnection
from infrastructure.persistence.conversation_repo import SQLiteConversationRepository
from infrastructure.persistence.medical_repo import SQLiteMedicalRepository
from infrastructure.persistence.migrations import run_migrations
from infrastructure.persistence.nutrition_facts_repo import SQLiteNutritionFactsRepository
from infrastructure.persistence.nutrition_facts_seed import seed_nutrition_facts
from infrastructure.persistence.nutrition_repo import SQLiteNutritionRepository
from infrastructure.persistence.profile_repo import SQLiteProfileRepository
from infrastructure.persistence.recipe_repo import SQLiteRecipeRepository
from infrastructure.persistence.user_repo import SQLiteUserRepository
from infrastructure.rag.medical_rag import MedicalRAG
from infrastructure.rag.recipe_rag import RecipeNutritionRAG

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
        self._medical_rag: MedicalRAG | None = None
        self._recipe_rag: RecipeNutritionRAG | None = None
        self._agent_llm = None
        self._intent_parser: IntentParser | None = None
        self._safety_filter: SafetyFilter | None = None
        self._image_detector = None

        self._initialized = False

    async def initialize(self) -> None:
        """One-time startup: run migrations, initialize RAG systems.

        Must be called before creating services or agents.
        """
        logger.info("Initializing ServiceFactory...")

        await run_migrations(self._connection)
        logger.info("Database migrations complete")

        # Reference food composition. Idempotent — a no-op once populated.
        nutrition_csv = self._config.data_dir / "cleaned_nutrition.csv"
        async with self._connection.acquire() as conn:
            rows = await seed_nutrition_facts(conn, nutrition_csv)
        logger.info("Nutrition facts table ready (%d foods)", rows)

        # ── Build shared LLM-backed components ONCE ──────────────────
        cfg = self._config
        self._intent_parser = IntentParser(
            provider=cfg.llm_provider,
            model=cfg.active_llm_model,
            ollama_base_url=cfg.ollama_base_url,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
        )
        self._safety_filter = SafetyFilter(
            provider=cfg.llm_provider,
            model=cfg.active_llm_model,
            ollama_base_url=cfg.ollama_base_url,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
        )
        self._agent_llm = self._build_agent_llm()
        self._image_detector = self._build_image_detector()
        logger.info("Shared LLM components initialized")

        # Shared cross-encoder reranker (built once, reused by both RAGs).
        # Loads its model lazily on first query and degrades gracefully if the
        # model can't be loaded, so this never blocks startup.
        reranker = self._build_reranker()

        # Shared embedding model — loaded ONCE here and injected into both RAGs
        # (a large model like BAAI/bge-m3 would otherwise be loaded twice).
        # Built in an executor to keep the heavy model load off the event loop.
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._build_embeddings)
        logger.info("Shared embeddings ready (%s)", cfg.embedding_model)

        # Initialize Medical RAG with configured provider
        self._medical_rag = MedicalRAG(
            folder_paths=[str(cfg.pdf_dir)],
            model_name=cfg.active_llm_model,
            vectorstore_path=str(cfg.medical_vectorstore_path),
            embedding_model=cfg.embedding_model,
            chunk_size=cfg.rag_chunk_size,
            chunk_overlap=cfg.rag_chunk_overlap,
            embeddings=embeddings,
            ollama_base_url=cfg.ollama_base_url,
            llm_provider=cfg.llm_provider,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
            use_hybrid=cfg.rag_use_hybrid,
            reranker=reranker,
            fetch_k=cfg.rag_fetch_k,
            rerank_top_n=cfg.rag_rerank_top_n,
        )
        await loop.run_in_executor(
            None, self._medical_rag.initialize, False,
        )
        logger.info("Medical RAG initialized (provider=%s)", cfg.llm_provider)

        # Initialize Recipe RAG with configured provider
        self._recipe_rag = RecipeNutritionRAG(
            data_folder=str(cfg.data_dir),
            model_name=cfg.active_llm_model,
            vectorstore_path=str(cfg.recipes_nutrition_vector_path),
            embedding_model=cfg.embedding_model,
            embeddings=embeddings,
            ollama_base_url=cfg.ollama_base_url,
            llm_provider=cfg.llm_provider,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
            use_hybrid=cfg.rag_use_hybrid,
            reranker=reranker,
        )
        await loop.run_in_executor(
            None, self._recipe_rag.initialize,
        )
        logger.info("Recipe RAG initialized (provider=%s)", cfg.llm_provider)

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
        facts_repo = SQLiteNutritionFactsRepository(self._connection)

        registry = ToolRegistry()
        registry.register(SearchRecipesTool(rec_service))
        registry.register(SaveRecipeTool(recipe_manager))
        registry.register(ShowRecipeTool())
        registry.register(AnalyzeImageTool(image_service))
        registry.register(GeneralChatTool())
        registry.register(NutritionStatusTool(nutrition_repo, medical_repo))
        registry.register(LookupNutritionTool(facts_repo))
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
        from infrastructure.llm.llm_builder import build_llm

        cfg = self._config
        logger.info(
            "Agent LLM: %s (%s)", cfg.llm_provider, cfg.active_llm_model,
        )
        return build_llm(
            provider=cfg.llm_provider,
            model=cfg.active_llm_model,
            temperature=0,
            ollama_base_url=cfg.ollama_base_url,
            openai_api_key=cfg.openai_api_key,
            groq_api_key=cfg.groq_api_key,
            chat_model=True,   # agent needs ChatModel for tool calling
        )

    def _build_embeddings(self):
        """Build the shared HuggingFace embedding model (called once during init)."""
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: %s", self._config.embedding_model)
        return HuggingFaceEmbeddings(
            model_name=self._config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_reranker(self):
        """Build the shared cross-encoder reranker, or None if disabled."""
        cfg = self._config
        if not cfg.rag_use_reranker:
            logger.info("RAG reranker disabled")
            return None
        from infrastructure.rag.retrieval import Reranker

        logger.info("RAG reranker: %s (top_n=%d)", cfg.rag_reranker_model, cfg.rag_rerank_top_n)
        return Reranker(
            model_name=cfg.rag_reranker_model,
            top_n=cfg.rag_rerank_top_n,
        )

    def _build_image_detector(self):
        """Build the configured ingredient detector (called once during init).

        Both backends satisfy IngredientDetectorPort, so nothing downstream
        cares which one is running.
        """
        if self._config.detector_backend == "yolo":
            from infrastructure.cnn.yolo_detector import YOLOIngredientDetector

            detector = YOLOIngredientDetector(
                weights=self._config.yolo_weights,
                confidence_threshold=self._config.yolo_confidence,
                device=self._config.yolo_device,
            )
            logger.info(
                "Image detector: YOLO ('%s', conf>=%.2f, device=%s)",
                self._config.yolo_weights,
                self._config.yolo_confidence,
                self._config.yolo_device,
            )
            return detector

        llava_model = self._config.cnn_model_path or "llava"
        detector = LLaVAIngredientDetector(
            ollama_base_url=self._config.ollama_base_url,
            model=llava_model,
        )
        logger.info(
            "Image detector: LLaVA ('%s') at %s",
            llava_model, self._config.ollama_base_url,
        )
        return detector

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "ServiceFactory not initialized. Call await factory.initialize() first."
            )
