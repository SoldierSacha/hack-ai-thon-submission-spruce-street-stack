import asyncio

from src.question_engine import QuestionEngine

# __name__ is the name of the current file; run is the "__main__" file because it's the file that's currently being run
if __name__ == "__main__":
    asyncio.run(QuestionEngine.main())

