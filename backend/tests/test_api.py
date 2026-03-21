"""
Integration tests for the PathForge API.
Run with: pytest tests/ -v
"""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_analyze_demo(client):
    res = await client.post(
        "/api/analyze",
        data={
            "role": "Software Engineer",
            "experience_level": "mid",
            "use_demo": "true",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert "pathway_nodes" in data
    assert "gap_skills" in data
    assert data["match_score"] >= 0
    assert data["total_training_days"] > 0


@pytest.mark.asyncio
async def test_chat_requires_session(client):
    res = await client.post(
        "/api/chat",
        json={
            "session_id": "nonexistent-session",
            "message": "What are my gaps?",
            "history": [],
        },
    )
    assert res.status_code == 200
    assert "expired" in res.json()["reply"].lower()


@pytest.mark.asyncio
async def test_pathway_not_found(client):
    res = await client.get("/api/pathway/fake-session-id")
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_full_flow(client):
    """Full end-to-end: analyze → chat → pathway fetch."""
    # Step 1: Analyze
    res = await client.post(
        "/api/analyze",
        data={
            "role": "Data Scientist",
            "experience_level": "beginner",
            "use_demo": "true",
        },
    )
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # Step 2: Chat
    chat_res = await client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "What are my skill gaps?",
            "history": [],
        },
    )
    assert chat_res.status_code == 200
    assert len(chat_res.json()["reply"]) > 0

    # Step 3: Fetch pathway
    pw_res = await client.get(f"/api/pathway/{session_id}")
    assert pw_res.status_code == 200
    assert pw_res.json()["session_id"] == session_id