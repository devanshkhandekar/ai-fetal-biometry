from __future__ import annotations
from dataclasses import dataclass
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.config import settings
from app.db import get_db
from app.models import Hospital, User, UserHospitalRole


bearer = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    id: str
    oidc_sub: str
    email: str | None
    hospital_id: str
    role: str


def _ensure_local_identity(db: Session, sub: str, email: str | None, role: str) -> CurrentUser:
    hospital = db.scalar(select(Hospital).where(Hospital.name == settings.default_hospital_name))
    if not hospital:
        hospital = Hospital(name=settings.default_hospital_name)
        db.add(hospital)
        db.flush()
    user = db.scalar(select(User).where(User.oidc_sub == sub))
    if not user:
        user = User(oidc_sub=sub, email=email)
        db.add(user)
        db.flush()
    link = db.scalar(
        select(UserHospitalRole).where(
            UserHospitalRole.user_id == user.id,
            UserHospitalRole.hospital_id == hospital.id,
        )
    )
    if not link:
        link = UserHospitalRole(user_id=user.id, hospital_id=hospital.id, role=role)
        db.add(link)
    elif role and link.role != role:
        link.role = role
    db.commit()
    return CurrentUser(user.id, user.oidc_sub, user.email, hospital.id, link.role)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> CurrentUser:
    if settings.auth_disabled:
        return _ensure_local_identity(db, "dev-local-user", "dev@local.test", "DOCTOR")
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = credentials.credentials
    try:
        jwks_client = PyJWKClient(settings.oidc_jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=settings.oidc_issuer,
            options={"verify_aud": False},
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc
    roles = claims.get("realm_access", {}).get("roles", [])
    role = next((r for r in ["SUPER_ADMIN", "HOSPITAL_ADMIN", "DOCTOR", "TECHNICIAN"] if r in roles), "DOCTOR")
    return _ensure_local_identity(db, claims["sub"], claims.get("email"), role)
