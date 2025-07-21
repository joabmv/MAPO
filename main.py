from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import convert_to_messages
from AGENTS.agents import supervisor


supervisor = supervisor.compile()

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


# for chunk in supervisor.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 #"content": "Otimize este portifólio AAPL, MSFT, GOOGL, NVDA utilizando o risk_parity optimizer, Markowitz optimizer, minima variancia optmizer e me mostre os resutados comparando ambos, utilize o periodo de 1y",
#                 "content": "Optmize this portifolio AAPL, MSFT, GOOGL, NVDA",
                
#             }
#         ]
#     },
# ):
#     pretty_print_messages(chunk, last_message=True)

# final_message_history = chunk["supervisor"]["messages"]

#===========================================================
conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = supervisor.invoke({"messages": conversation_history})
    print(f'IA: {result['messages'][-1].content}')
    conversation_history = result["messages"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")
