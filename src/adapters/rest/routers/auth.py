"""Auth endpoints: register and login."""

from fastapi import APIRouter, Depends, HTTPException, status

from factory import ServiceFactory
from domain.exceptions import AuthenticationError, DuplicateLoginError
from application.dto import RegisterRequest, LoginRequest
from adapters.rest.dependencies import get_factory
from adapters.rest.schemas import RegisterBody, LoginBody, TokenResponse, RefreshBody

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    body: RegisterBody,
    factory: ServiceFactory = Depends(get_factory),
):
    auth_service = factory.create_authentication_service()
    try:
        token = await auth_service.register(RegisterRequest(
            login=body.login,
            password=body.password,
            name=body.name,
            surname=body.surname,
            age=body.age,
            gender=body.gender,
            caretaker=body.caretaker,
            health_condition=body.health_condition,
        ))
    except DuplicateLoginError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    # Persist health conditions immediately so load_user_context() picks them
    # up on first login without requiring a prior chat interaction.
    if body.health_condition:
        profile_svc = factory.create_profile_service()
        await profile_svc.save_initial_profile(token.user_id, body.health_condition)

    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        user_id=token.user_id,
        role=token.role,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginBody,
    factory: ServiceFactory = Depends(get_factory),
):
    auth_service = factory.create_authentication_service()
    try:
        token = await auth_service.login(LoginRequest(
            login=body.login,
            password=body.password,
        ))
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        user_id=token.user_id,
        role=token.role,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshBody,
    factory: ServiceFactory = Depends(get_factory),
):
    """Re-issue a new JWT using an existing (possibly expired) token.

    The Flutter app should call this when the stored token is close to
    expiry, avoiding a full re-login.
    """
    auth_service = factory.create_authentication_service()
    try:
        token = await auth_service.refresh_token(body.token)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        user_id=token.user_id,
        role=token.role,
    )
