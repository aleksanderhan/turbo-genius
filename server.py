import argparse
import asyncio
from fastapi import FastAPI, WebSocket, Depends
from sqlalchemy.orm import Session as DBSession
from llama_cpp import Llama

from session import Session, SessionManager, SessionDB, get_db

app = FastAPI()
session_manager = SessionManager()

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default='models/model.gguf')
parser.add_argument('--port', action='store', type=int, default=8000)
parser.add_argument('--n_ctx', action='store', type=int, default=4096)
parser.add_argument('--n_gpu_layers', action='store', type=int, default=0)
parser.add_argument('--n_threads', action='store', type=int, default=8)
args = parser.parse_args()

llm = Llama(
    model_path=args.model,
    n_ctx=args.n_ctx,
    n_gpu_layers=args.n_gpu_layers,
    n_threads=args.n_threads,
    verbose=False,
)


def make_title(session: Session):
    user_messages = [m['content'] for m in session.get_messages() if m['role'] == 'user']
    if not user_messages:
        return 'New session'
    words = user_messages[0].strip().split()
    return ' '.join(words[:8]) or 'New session'


def make_messages(session: Session):
    messages = session.get_messages()
    if len(messages) > 20:
        session.messages = session.messages[-20:]
        messages = session.get_messages()
    return messages


async def generate_response(messages):
    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
    )
    for chunk in stream:
        choices = chunk.get('choices', [])
        if not choices:
            continue
        delta = choices[0].get('delta', {})
        token = delta.get('content')
        if token:
            yield token


@app.websocket('/stream/{session_id}')
async def stream(websocket: WebSocket, session_id: int, db: DBSession = Depends(get_db)):
    await websocket.accept()
    message = await websocket.receive_text()
    session = session_manager.get_session(session_id, db)

    session.add_user_message(message)
    session_manager.save_session(session, db)

    messages = make_messages(session)
    completion = ''
    try:
        async for token in generate_response(messages):
            completion += token
            await websocket.send_text(token)
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        session.add_assistant_message(completion)
        session_manager.save_session(session, db)
        await websocket.close()


@app.get('/session')
async def get_session(db: DBSession = Depends(get_db)):
    session = session_manager.get_new_session(db)
    return session.id


@app.get('/session/{session_id}')
async def get_session_by_id(session_id: int, db: DBSession = Depends(get_db)):
    session = session_manager.get_session(session_id, db)
    return session


@app.get('/session-list')
async def get_session_list(db: DBSession = Depends(get_db)):
    sessions = session_manager.get_session_list(db)
    return sessions


@app.delete('/session/{session_id}')
async def delete_session(session_id: int, db: DBSession = Depends(get_db)):
    session_manager.remove_session(session_id, db)
    db.commit()
    return


@app.get('/session/{session_id}/title')
async def get_session_title(session_id: int, db: DBSession = Depends(get_db)):
    session = session_manager.get_session(session_id, db)
    session.title = make_title(session)
    db_session = db.query(SessionDB).filter(SessionDB.id == session.id).first()
    db_session.title = session.title
    db.add(db_session)
    db.commit()
    return session.title


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=args.port)
