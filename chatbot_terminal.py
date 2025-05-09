from chatbot_core import build_qa_chain

qa_chain = build_qa_chain("ucf_rules.pdf") # Choose the PDF file to use

print("🧠 PDF-Chatbot started! Enter ‘exit’ to quit.")

while True:
    query = input("\n❓ Your questions: ")
    # Breaks the loop if the user types 'exit' or 'quit'
    if query.lower() in ["exit", "quit"]:
        print("👋 Chat finished.")
        break

    # Get the answer from the QA chain (LLM + Retriever) and prints the answer to the terminal
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    print("\n💬 Answer:", result["answer"])
    chat_history.append((query, result["answer"])) #Saves the Q&A pair in the chat history
    print("\n🔍 Source – Document snippet:") #Shows a snippet from the source document that is used
    print(result["source_documents"][0].page_content[:300])