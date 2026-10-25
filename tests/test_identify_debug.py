import asyncio
from app.api.routes import identify
from app.services.recommender import recommender
from fastapi import UploadFile
import io

async def test_audio_identify_and_sorting():
    recommender.load()
    file = UploadFile(filename="test.webm", file=io.BytesIO(b"fake audio data 1234567890"))
    res = await identify(file)
    assert res.identified is not None
    assert len(res.recommendations) > 0
    scores = [r.matchScore for r in res.recommendations]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores must be descending"
    print("Test passed successfully! Found", len(res.recommendations), "strictly sorted tracks.")

if __name__ == "__main__":
    asyncio.run(test_audio_identify_and_sorting())
