# telegram_mongo_poller.py

import asyncio
from datetime import datetime, timedelta
from telethon import TelegramClient, events
from telethon.tl.types import (
    Chat, Channel, MessageMediaPoll, MessageMediaPhoto,
    DocumentAttributeFilename, UpdateMessagePoll, UpdateMessagePollVote
)
from pymongo import MongoClient
from gridfs import GridFS

# ===========================
# 🔑 Replace with your values
# ===========================
API_ID = 27239431       # <-- your API ID
API_HASH = "e0c2dd19e9ea0da9a3a7a102393a1b69"
SESSION_NAME = "poll_session"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "telegram_data"

# ===========================
# 📦 MongoDB Setup
# ===========================
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db["messages"]
fs = GridFS(db)

selected_chats = []
LISTENING = True   # <-- controls if bot is in listening mode

# ===========================
# 📌 Poll Parser
# ===========================
def parse_poll(msg):
    poll = msg.media.poll
    poll_results = msg.media.results
    answers_map = {a.option: a.text for a in poll.answers}

    poll_data = {
        "question": str(poll.question),   # 🔥 force string instead of TextWithEntities
        "total_voters": getattr(poll_results, "total_voters", None),
        "closed": poll.closed,
        "answers": []
    }

    if poll_results and poll_results.results:
        for r in poll_results.results:
            poll_data["answers"].append({
                "text": answers_map.get(r.option, None),
                "voters": r.voters,
                "option": r.option.decode("utf-8") if r.option else None
            })
    return poll_data

# ===========================
# 📌 Save Messages
# ===========================
async def save_message(msg, chat_name=None, sender_name=None):
    doc = {
        "message_id": msg.id,
        "chat_id": msg.chat_id,
        "chat_name": chat_name,
        "sender": sender_name,
        "date": msg.date.isoformat() if msg.date else None,
        "text": msg.text,
        "media_type": None,
        "file_id": None,
        "poll_data": None
    }

    # Polls
    if isinstance(msg.media, MessageMediaPoll):
        doc["media_type"] = "poll"
        doc["poll_data"] = parse_poll(msg)

    # Photos
    elif isinstance(msg.media, MessageMediaPhoto):
        file_bytes = await msg.download_media(bytes)
        file_id = f"photo_{msg.id}.jpg"
        fs.put(file_bytes, filename=file_id, metadata={"chat": chat_name})
        doc["media_type"] = "photo"
        doc["file_id"] = file_id

    # Documents (video, audio, files)
    elif msg.document:
        mime = msg.document.mime_type
        file_name = f"file_{msg.id}"
        for attr in msg.document.attributes:
            if isinstance(attr, DocumentAttributeFilename):
                file_name = attr.file_name
        file_bytes = await msg.download_media(bytes)
        fs.put(file_bytes, filename=file_name, metadata={"mime": mime, "chat": chat_name})
        doc["media_type"] = "document"
        doc["file_id"] = file_name
        doc["mime_type"] = mime

    collection.update_one(
        {"message_id": msg.id, "chat_id": msg.chat_id},
        {"$set": doc},
        upsert=True
    )
    print(f"[💾] Saved {doc['media_type'] or 'text'} from {chat_name}")

# ===========================
# 📌 Update Polls
# ===========================
async def update_poll(update):
    poll_id = update.poll_id
    print(f"🔄 Poll updated: {poll_id}")

    msg_doc = collection.find_one({"poll_data": {"$exists": True}})
    if not msg_doc:
        return

    msg = await client.get_messages(msg_doc["chat_id"], ids=msg_doc["message_id"])
    if msg and isinstance(msg.media, MessageMediaPoll):
        poll_data = parse_poll(msg)
        collection.update_one(
            {"message_id": msg.id, "chat_id": msg.chat_id},
            {"$set": {"poll_data": poll_data}},
            upsert=True
        )
        print(f"[✅] Poll {poll_id} updated in MongoDB.")

# ===========================
# 📌 Choose Chats
# ===========================
async def choose_chats():
    global selected_chats
    dialogs = await client.get_dialogs()

    print("\n📌 Available chats:\n")
    for i, dialog in enumerate(dialogs, start=1):
        entity = dialog.entity
        name = entity.title if isinstance(entity, (Chat, Channel)) else getattr(entity, "first_name", None) or entity.username
        print(f"{i}. {name}")

    choice = input("\n👉 Enter chat numbers to track (comma separated): ")
    choices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]

    selected_chats = []  # reset list before adding
    for idx in choices:
        if 1 <= idx <= len(dialogs):
            selected_chats.append(dialogs[idx - 1].id)

    print("\n✅ Tracking started for selected chats...\n")

# ===========================
# 📌 Fetch Old Messages
# ===========================
async def fetch_old_messages():
    print("\nChoose fetch mode for old messages:")
    print("1. Fetch ALL messages")
    print("2. Fetch last N messages")
    print("3. Fetch messages from last X days")
    choice = input("Enter choice (1/2/3): ")

    limit, min_date = None, None
    if choice == "2":
        n = int(input("Enter N (e.g. 500): "))
        limit = n
    elif choice == "3":
        days = int(input("Enter number of days (e.g. 30): "))
        min_date = datetime.utcnow() - timedelta(days=days)

    # Fetch old messages
    if choice in ["1", "2", "3"]:
        for chat_id in selected_chats:
            entity = await client.get_entity(chat_id)
            chat_name = getattr(entity, "title", getattr(entity, "first_name", "Unknown"))
            print(f"📂 Fetching history for {chat_name} ({chat_id})")

            async for msg in client.iter_messages(chat_id, limit=limit):
                if min_date and msg.date < min_date:
                    continue
                sender = await msg.get_sender()
                sender_name = getattr(sender, "username", getattr(sender, "first_name", "Unknown"))
                await save_message(msg, chat_name, sender_name)

        print("✅ Finished fetching old messages.")

# ===========================
# 🚀 Main Runner
# ===========================
async def main():
    global client, LISTENING
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    print("✅ Telegram client started!")

    # First time setup
    await choose_chats()
    await fetch_old_messages()

    # Handlers
    @client.on(events.NewMessage)
    async def handler(event):
        global LISTENING
        # Only process /menu if message is from you
        me = await client.get_me()
        if event.sender_id == me.id and event.raw_text.strip().lower() == "/menu":
            LISTENING = False
            print("\n[🔄] Returning to menu...\n")
            await choose_chats()
            await fetch_old_messages()
            LISTENING = True
            await event.reply("📌 Menu reloaded. Monitoring updated chats.")
            return

        if LISTENING and (event.chat_id in selected_chats):
            sender = await event.get_sender()
            chat = await event.get_chat()
            sender_name = getattr(sender, "username", getattr(sender, "first_name", "Unknown"))
            chat_name = getattr(chat, "title", getattr(sender, "username", "Private Chat"))
            await save_message(event.message, chat_name, sender_name)

    @client.on(events.Raw)
    async def raw_handler(event):
        if isinstance(event, (UpdateMessagePoll, UpdateMessagePollVote)):
            await update_poll(event)

    print("📡 Listening for new messages & poll updates...")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
