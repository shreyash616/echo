"""
Verifies that unrecognized audio clips fall back gracefully to neural audio analysis rather than crashing with 500.
"""
import asyncio
from app.api.routes import identify
from app.services.recommender import recommender
from fastapi import UploadFile
import io

async def test_audio_identify_fallback():
    recommender.load()
    file = UploadFile(filename="test.webm", file=io.BytesIO(b"fake audio data 1234567890"))
    res = await identify(file)
    assert res is not None
    assert res.identified is not None
    scores = [r.matchScore for r in res.recommendations if r.matchScore is not None]
    if scores:
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores must be descending"
    print("Test passed: fallback correctly returned without 500 error.")

if __name__ == "__main__":
    asyncio.run(test_audio_identify_fallback())
